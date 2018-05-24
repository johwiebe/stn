#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Deterministic model of STN with degradation. Based on Biondi et al 2017.
'''
import pyomo.environ as pyomo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
import pandas as pd
import dill
import collections
from deg import degradationModel, calc_p_fail
from blocks import (blockScheduling, blockSchedulingRobust,
                    blockPlanning, blockPlanningRobust)


class stnModel(object):
    """Deterministic model of the STN."""
    def __init__(self, stn=None):
        self.Demand = {}            # demand for products
        if stn is None:
            self.stn = stnStruct()  # contains STN architecture
        else:
            self.stn = stn
        self.m_list = []
        self.gapmin = 100
        self.gapmax = 0
        self.gapmean = 0
        self.alpha = 0.5
        self.rid = 0
        self.prefix = ''
        self.rdir = 'results'

    def solve(self, T_list, periods=1, solver='cplex',
              solverparams=None,
              save=False, trace=False, gantt=True, **kwargs):
        """
        Solves the model

            T_list: []
            periods: number of rolling horizon periods
            solver: specifies which solver to use
            prefix: added to all file names
            rdir: directory for result files
            solverparams: dictionary of solver parameters
            save: save results as .pyomo?
            trace: generate trace?
            gantt: generate gantt graphs?

        """
        # Initialize solver and set parameters
        ts = time.time()
        self.solver = pyomo.SolverFactory(solver)
        if solverparams is not None:
            for key, value in solverparams.items():
                self.solver.options[key] = value

        # Rolling horizon
        for period in range(0, periods):
            if periods > 1:
                rolling = True
            else:
                rolling = False
            # Build model
            self.build(T_list, period=period, rolling=rolling, **kwargs)
            logfile = self.prfx + "STN.log"
            # Solve model
            results = self.solver.solve(self.model,
                                        tee=True,
                                        keepfiles=True,
                                        symbolic_solver_labels=True,
                                        logfile=logfile)
            results.write()
            # Check if solver exited normally
            if ((results.solver.status == SolverStatus.ok) and
                (results.solver.termination_condition ==
                 TerminationCondition.optimal or
                 results.solver.termination_condition ==
                 TerminationCondition.maxTimeLimit)):
                # Calculate MIP Gap
                obj = self.model.Obj()
                gap = self.solver._gap
                if gap is None:
                    gap = 0.0
                self.gapmin = min(self.gapmin,
                                  gap/obj*100)
                self.gapmax = max(self.gapmax,
                                  gap/obj*100)
                self.gapmean = (self.gapmean
                                * period/(period+1)
                                + (1 - period/(period + 1))
                                * gap/obj*100)
                # Save results
                if save:
                    with open(self.prfx+'output.txt', 'w') as f:
                        f.write("STN Output:")
                        self.model.display(ostream=f)
                    with open(self.prfx+'STN.pyomo', 'wb') as dill_file:
                        dill.dump(self.model, dill_file)
                if gantt:
                    self.sb.gantt()
                    self.pb.gantt()
                if trace:
                    self.sb.trace()
                    self.pb.trace()
                if periods > 1:
                    self.transfer_next_period(**kwargs)
                # Add current model to list
                self.m_list.append(self.model)
            else:
                break
        self.ttot = time.time() - ts

    def build(self, T_list, objective="terminal", period=None, alpha=0.5,
              extend=False, rdir='results', prefix='', rolling=False,
              rid=None, **kwargs):
        """Build STN model."""
        assert period is not None
        self.rdir = rdir
        self.prefix = prefix
        if rid is not None:
            self.rid = rid
        else:
            try:
                df = pd.read_pickle(self.rdir+"/"+self.prefix+"results.pkl")
                self.rid = max(df["id"]) + 1
            except IOError:
                pass
        self.prfx = self.rdir + "/" + self.prefix + str(self.rid)
        if rolling:
            self.prfx += "_" + str(period)
        self.model = pyomo.ConcreteModel()
        m = self.model
        stn = self.stn
        m.cons = pyomo.ConstraintList()
        m.ptransfer = pyomo.Var(stn.tasks, stn.units, stn.opmodes,
                                domain=pyomo.NonNegativeReals)
        m.Btransfer = pyomo.Var(stn.tasks, stn.units, stn.opmodes,
                                domain=pyomo.NonNegativeReals)
        m.tautransfer = pyomo.Var(stn.units, domain=pyomo.NonNegativeReals)
        m.Dslack = pyomo.Var(stn.states, domain=pyomo.NonNegativeReals)
        m.TotSlack = pyomo.Var(domain=pyomo.NonNegativeReals)

        # replace D by Dmax if alpha != self.alpha
        if alpha != self.alpha:
            for j in stn.units:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        tm = i + "-" + k
                        p = stn.p[i, j, k]
                        D = stn.deg[j].get_mu(tm, p)
                        eps = stn.deg[j].get_eps(alpha, tm, p)
                        # X = stn.deg[j].get_quantile(alpha, tm, p)
                        # stn.D[i, j, k] = 2*D - X
                        stn.D[i, j, k] = D*(1 + eps)
            self.alpha = alpha

        # scheduling and planning block
        Ts = T_list[0]
        dTs = T_list[1]
        Tp = T_list[2]
        dTp = T_list[3]
        Ts_start = period * Ts
        Tp_start = (period + 1) * Ts
        Ts = Ts_start + Ts
        if extend:
            Tp = Ts_start + Tp
        self.add_blocks([Ts_start, Ts, dTs], [Tp_start, Tp, dTp],
                        objective=objective, **kwargs)

        # add continuity constraints to model
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints(**kwargs)

        # add objective function to model
        self.add_objective()

    def add_blocks(self, TIMEs, TIMEp, **kwargs):
        """Add scheduling and planning block to model."""
        stn = self.stn
        m = self.model
        m.sb = pyomo.Block()
        self.sb = blockScheduling(stn, TIMEs,
                                  self.Demand, prfx=self.prfx, **kwargs)
        self.sb.define_block(m.sb, **kwargs)
        m.pb = pyomo.Block()
        self.pb = blockPlanning(stn, TIMEp,
                                self.Demand, prfx=self.prfx, **kwargs)
        self.pb.define_block(m.pb, **kwargs)

    def add_unit_constraints(self):
        """Add unit allocation continuity constraints to model."""
        m = self.model
        stn = self.stn
        # continuity of allocation
        for j in stn.units:
            rhs = 0
            for i in stn.I[j]:
                for k in stn.O[j]:
                    # Add processing time of tasks started in scheduling
                    # horizon
                    rhs2 = 0
                    rhs3 = 0
                    for tprime in self.sb.TIME[self.sb.TIME
                                               >= self.sb.T
                                               - stn.p[i, j, k]
                                               + self.sb.dT]:
                        rhs2 += (m.sb.W[i, j, k, tprime]
                                 * (stn.p[i, j, k]
                                    - (self.sb.T - tprime)))
                        rhs += rhs2
                        rhs3 += m.sb.B[i, j, k, tprime]
                    m.cons.add(m.ptransfer[i, j, k] == rhs2)
                    m.cons.add(m.Btransfer[i, j, k] == rhs3)
            # Add time for maintenance started in scheduling horizon
            rhs2 = 0
            for tprime in self.sb.TIME[(self.sb.TIME
                                        >= self.sb.T
                                        - stn.tau[j]
                                        + self.sb.dT)]:
                rhs2 += m.sb.M[j, tprime]*(stn.tau[j] - (self.sb.T - tprime))
                rhs += rhs2
            m.cons.add(m.pb.Ntransfer[j] == rhs)  # TODO: should this time?
            m.cons.add(m.tautransfer[j] == rhs2)

    def add_state_constraints(self):
        """Add state continuity constraints to model."""
        m = self.model
        stn = self.stn
        for s in stn.states:
            # Calculate states at end of scheduling horizon
            rhs = m.sb.S[s, self.sb.T - self.sb.dT]
            for i in stn.T_[s]:
                for j in stn.K[i]:
                    for k in stn.O[j]:
                        rhs += (stn.rho_[(i, s)]
                                * m.sb.B[i, j, k,
                                         self.sb.T
                                         - stn.p[i, j, k]])
            # Subtract demand from last scheduling period
            if (s, self.sb.TIME[0]) in self.Demand:
                rhs -= self.Demand[s, self.sb.TIME[0]]
                rhs += m.Dslack[s]
            m.cons.add(m.sb.Sfin[s] == rhs)
            m.cons.add(0 <= m.sb.Sfin[s] <= stn.C[s])
            # Calculate amounts transfered into planning period
            for i in stn.T_[s]:
                for j in stn.K[i]:
                    for k in stn.O[j]:
                        for tprime in self.sb.TIME[self.sb.TIME
                                                   >= self.sb.T
                                                   - stn.p[i, j, k]
                                                   + self.sb.dT]:
                            rhs += stn.rho_[(i, s)] * m.sb.B[i, j, k, tprime]
            m.cons.add(m.pb.Stransfer[s] == rhs)

    def add_deg_constraints(self, **kwargs):
        """Add residual life continuity constraints to model."""
        stn = self.stn
        m = self.model
        for j in stn.units:
            m.cons.add(m.pb.Rtransfer[j] == m.sb.R[j, self.sb.T - self.sb.dT])

    def add_objective(self):
        """Add objective function."""
        m = self.model
        stn = self.stn
        totslack = 0
        for s in stn.states:
            totslack += m.Dslack[s]
            for t in m.sb.TIME:
                totslack += m.sb.Sslack[s, t]
            for t in m.pb.TIME:
                totslack += m.pb.Dslack[s, t]
        m.cons.add(m.TotSlack == totslack)
        m.Obj = pyomo.Objective(expr=m.sb.Cost          # cost scheduling blk
                                + m.pb.Cost             # cost planning blk
                                + m.TotSlack*10000,     # penalize slack vars
                                sense=pyomo.minimize)

    def demand(self, state, time, Demand):
        """Add demand to model."""
        self.Demand[state, time] = Demand

    def uncertainty(self, alpha):
        """Set uncertainty set size parameter."""
        self.alpha = alpha

    def transfer_next_period(self, **kwargs):
        """
        Transfer results from previous scheduling period to next (rolling
        horizon).
        """
        m = self.model
        stn = self.stn

        for s in stn.states:
            stn.init[s] = m.sb.Sfin[s]()
        for j in stn.units:
            for i in stn.I[j]:
                for k in stn.O[j]:
                    stn.pinit[i, j, k] = m.ptransfer[i, j, k]()
                    stn.Binit[i, j, k] = m.Btransfer[i, j, k]()
                    stn.tauinit[j] = m.tautransfer[j]()
                    if m.ptransfer[i, j, k]() < self.sb.dT/2:
                        stn.pinit[i, j, k] = 0
                        stn.Binit[i, j, k] = 0
                    if m.tautransfer[j]() < self.sb.dT/2:
                        stn.tauinit[j] = 0
            stn.Rinit[j] = m.sb.R[j, self.sb.T - self.sb.dT]()

    def loadres(self, prefix="", f="STN.pyomo", periods=0):
        if periods == 0:
            with open(prefix + f, 'rb') as dill_file:
                self.model = dill.load(dill_file)
        else:
            for period in range(0, periods):
                with open(prefix + str(period) + f, 'rb') as dill_file:
                    m = dill.load(dill_file)
                    self.m_list.append(m)
            self.model = m
            self.sb.b = m.sb
            self.pb.b = m.pb

    def get_gap(self):
        return self.gapmax, self.gapmean, self.gapmin

    def calc_p_fail(self, units=None, TP=None, periods=0, pb=True, save=True):
        assert TP is not None
        if units is None:
            units = self.stn.units
        elif type(units) == str:
            units = set([units])
        df = pd.DataFrame(columns=units)
        for j in units:
            df[j] = calc_p_fail(self, j, self.alpha, TP, pb=pb,
                                periods=periods)
        df["alpha"] = self.alpha
        df["id"] = self.rid
        if save:
            try:
                df2 = pd.read_pickle(self.rdir+"/"+self.prefix+"pfail.pkl")
                df2 = df2.append(df)
            except IOError:
                df2 = df
            df2.to_pickle(self.rdir+"/"+self.prefix+"pfail.pkl")
            df2.to_csv(self.rdir+"/"+self.prefix+"pfail.csv")
        return df

    def check_for_task(self, model, j, t):
        b = model.sb
        stn = self.stn
        tend = t
        for i in stn.I[j]:
            for k in stn.O[j]:
                W = b.W[i, j, k, t]()
                if W > 0.5:
                    tend = t + stn.p[i, j, k]
                    return [i, k, tend]
        i = "None"
        k = "None"
        M = b.M[j, t]()
        if M > 0.5:
            tend = t + stn.tau[j]
            i = "M"
            k = "M"
        return [i, k, tend]

    def get_unit_profile(self, j, full=False):
        cols = ["period", "time", "unit", "task", "mode"]
        prods = self.stn.products
        demand = []
        for p in prods:
            cols.append(p)
            demand.append(0)
        profile = pd.DataFrame(columns=cols)
        tend = 0
        for period, m in enumerate(self.m_list):
            for t in m.sb.TIME:
                tend_old = tend
                if t >= tend:
                    [i, k, tend] = self.check_for_task(m, j, t)
                line = [period, t, j, i, k]
                for n, p in enumerate(prods):
                    if (p, t) in self.Demand:
                        demand[n] = self.Demand[(p, t)]
                line += demand
                if full or t >= tend_old:
                    profile = profile.append(pd.Series(line,
                                                       index=cols),
                                             ignore_index=True)
        return profile

    def get_hist(self, j):
        stn = self.stn
        tms = ["None-None", "M-M"]
        for i in stn.I[j]:
            for k in stn.O[j]:
                tms.append(i + "-" + k)
        prof = self.get_unit_profile(j)
        c = collections.Counter(prof["task"] + "-" + prof["mode"])
        df = pd.DataFrame.from_dict({tm: [c[tm]] for tm in tms})
        df["alpha"] = self.alpha
        df["rid"] = self.rid
        return df

    def get_production_targets(self):
        m = self.model
        stn = self.stn
        cols = ["time"]
        prods = stn.products
        for p in prods:
            cols.append(p)
        df = pd.DataFrame(columns=cols)
        for t in m.pb.TIME:
            target = []
            for p in prods:
                rhs = 0
                for i in stn.T_[p]:
                    for j in stn.K[i]:
                        rhs += stn.rho_[(i, p)]*m.pb.A[i, j, t]()
                target.append(rhs)
            line = [t] + target
            df = df.append(pd.Series(line, index=cols),
                           ignore_index=True)
        return df

    def eval(self, save=True, periods=None, **kwargs):
        if periods is None:
            periods = len(self.m_list)
        cols = ["id", "alpha", "CostStorage", "CostMaintenance",
                "CostMainenanceFinal", "Cost", "slack", "ttot",
                "gapmin", "gapmean", "gapmax"]
        cols += self.stn.products
        units = [j for j in self.stn.units]
        dem = [0 for p in self.stn.products]
        cols += units
        cost_storage = 0
        cost_maint = 0
        cost = 0
        slack = 0
        last_i = 0
        for i, m in enumerate(self.m_list):
            if i < periods:
                cost_storage += m.sb.CostStorage()
                cost_maint += m.sb.CostMaintenance()
                slack += m.TotSlack()
                for n, p in enumerate(self.stn.products):
                    t = i*self.pb.dT
                    if (p, t) in self.Demand:
                        dem[n] += self.Demand[(p, t)]
                last_i = i
        if periods <= len(self.m_list):
            last_sb = self.m_list[last_i].sb
            cost_maint_final = self.sb.get_cost_maint_terminal(last_sb)
        else:
            for i in range(len(self.m_list), periods):
                cost_storage += m.pb.CostStorage[i*self.pb.dT]()
                cost_maint += m.pb.CostMaintenance[i*self.pb.dT]()
            cost_maint_final = self.pb.get_cost_maint_terminal(periods)
        cost += (cost_storage + cost_maint
                 + cost_maint_final)
        pf = self.calc_p_fail(save=save, **kwargs)
        pl = []
        for j in units:
            pl.append(max(pf[j]))
        df = pd.DataFrame([[self.rid, self.alpha, cost_storage,
                            cost_maint, cost_maint_final, cost,
                            slack, self.ttot,
                            self.gapmin, self.gapmean, self.gapmax]
                           + dem + pl],
                          columns=cols)
        if save:
            try:
                df2 = pd.read_pickle(self.rdir+"/"+self.prefix+"results.pkl")
                df2 = df2.append(df)
            except IOError:
                df2 = df
            df2.to_pickle(self.rdir+"/"+self.prefix+"results.pkl")
            df2.to_csv(self.rdir+"/"+self.prefix+"results.csv")
        return df


class stnModelRobust(stnModel):
    """Robust model for STN."""
    def __init__(self, stn=None):
        super().__init__(stn)

    def build(self, T_list, period=None, tindexed=True,
              alpha=0.5, **kwargs):
        assert period is not None
        stn = self.stn
        self.alpha = alpha
        for j in stn.units:
            for i in stn.I[j]:
                for k in stn.O[j]:
                    tm = i + "-" + k
                    p = stn.p[i, j, k]
                    stn.eps[i, j, k] = stn.deg[j].get_eps(alpha, tm, p)
        super().build(T_list, alpha=alpha, period=period,
                      tindexed=tindexed, **kwargs)

    def add_blocks(self, TIMEs, TIMEp, decisionrule="continuous", **kwargs):
        stn = self.stn
        m = self.model
        available_rules = ["continuous", "integer"]

        assert decisionrule in available_rules, ("Unknown decision rule %s" %
                                                 decisionrule)

        # scheduling and planning block
        m.sb = pyomo.Block()
        m.pb = pyomo.Block()
        self.sb = blockSchedulingRobust(stn,
                                        np.array([t for t in TIMEs]),
                                        self.Demand,
                                        decisionrule=decisionrule,
                                        prfx=self.prfx,
                                        **kwargs)
        self.sb.define_block(m.sb, decisionrule=decisionrule, **kwargs)
        self.pb = blockPlanningRobust(stn,
                                      np.array([t for t in TIMEp]),
                                      self.Demand,
                                      decisionrule=decisionrule,
                                      prfx=self.prfx,
                                      **kwargs)
        self.pb.define_block(m.pb, decisionrule=decisionrule, **kwargs)

    def add_deg_constraints(self, tindexed=None, **kwargs):
        assert tindexed is not None
        stn = self.stn
        m = self.model
        for j in stn.units:
            m.cons.add(m.pb.R0transfer[j] == m.sb.R0[j, self.sb.T -
                                                     self.sb.dT])
            for i in stn.I[j]:
                for k in stn.O[j]:
                    rhs = 0
                    if tindexed:
                        for t in self.sb.TIME:
                            rhs += m.sb.Rc[j, self.sb.T - self.sb.dT, i, k, t]
                    else:
                        rhs += m.sb.Rc[j, self.sb.T - self.sb.dT, i, k]
                    m.cons.add(m.pb.Rctransfer[j, i, k] == rhs)

    def transfer_next_period(self, deg_continuity="max", **kwargs):
        """ Transfer results from end of current scheduling period. """
        stn = self.stn
        m = self.model
        super().transfer_next_period(**kwargs)
        for j in stn.units:
            if deg_continuity == "max":
                stn.Rinit[j] = m.sb.Rmax[j, self.sb.T - self.sb.dT]()


class stnStruct(object):
    def __init__(self):
        # simulation objects
        self.states = set()         # set of state names
        self.tasks = set()          # set of task names
        self.units = set()          # set of unit names
        self.opmodes = set()        # set of operating mode names
        self.products = []

        # constants
        self.U = 100                # big U
        # self.eps = 0.0            # Maximum deviation for uncertain D's FIX!
        # self.alpha = 0.5            # uncertainty set size parameter

        # dictionaries indexed by task name
        self.S = {}                 # sets of states feeding each task (inputs)
        self.S_ = {}                # sets of states fed by each task (outputs)
        self.K = {}                 # sets of units capable of each task
        # self.p = {}                 # task durations

        # dictionaries indexed by state name
        self.T = {}                 # tasks fed from each state (task output)
        self.T_ = {}                # tasks feeding each state (task inputs)
        self.C = {}                 # capacity of each task
        self.init = {}              # initial level
        self.price = {}             # prices of each state
        self.scost = {}

        # dictionary indexed by unit
        self.I = {}                 # noqa set of tasks performed by each unit
        self.O = {}                 # noqa sets of op modes for each unit
        self.tau = {}               # time taken for maintenance on each unit
        self.a = {}                 # fixed maintenance cost for each unit
        self.b = {}                 # maintenance discount for each unit
        self.tauinit = {}           # initial time of maintenance left
        self.deg = {}               # degradation models

        # dictionaries indexed by (task, state)
        self.rho = {}               # input feed fractions
        self.rho_ = {}              # output product dispositions
        self.P = {}                 # time to finish output from task to state

        # characterization of units indexed by (task, unit, operating mode)
        self.Bmax = {}              # max capacity of unit j for task i
        self.Bmin = {}              # minimum capacity of unit j for task i
        self.cost = {}
        self.vcost = {}
        self.Rmax = {}
        self.Rinit = {}
        self.Rinit0 = {}
        self.eps = {}

        # characterization of units indexed by (task, unit, operating mode)
        self.p = {}                 # task duration
        self.D = {}                 # wear coefficient
        self.pinit = {}             # initial processing time left
        self.Binit = {}             # initial amount being processed

        # dictionaries indexed by (task,task)
        self.changeoverTime = {}    # time required for task1 -> task2

        # dictionaries indexed by operating mode
        self.modeorder = {}

    # defines states as .state(name, capacity, init)
    def state(self, name, capacity=float('inf'), init=0, price=0, scost=0,
              prod=False):
        self.states.add(name)       # add to the set of states
        self.C[name] = capacity     # state capacity
        self.init[name] = init      # state initial value
        self.T[name] = set()        # tasks which feed this state (inputs)
        self.T_[name] = set()       # tasks fed from this state (outputs)
        self.price[name] = price    # per unit price of each state
        self.scost[name] = scost    # storage cost per (planning) interval
        if prod:
            if name not in self.products:
                self.products.append(name)

    def task(self, name):
        self.tasks.add(name)        # add to set of tasks
        self.S[name] = set()        # states which feed this task (inputs)
        self.S_[name] = set()       # states fed by this task (outputs)
        self.p[name] = 0            # completion time for this task
        self.K[name] = set()

    def stArc(self, state, task, rho=1):
        if state not in self.states:
            self.state(state)
        if task not in self.tasks:
            self.task(task)
        self.S[task].add(state)
        self.rho[(task, state)] = rho
        self.T[state].add(task)

    def tsArc(self, task, state, rho=1, dur=0):
        if state not in self.states:
            self.state(state)
        if task not in self.tasks:
            self.task(task)
        self.S_[task].add(state)
        self.T_[state].add(task)
        self.rho_[(task, state)] = rho
        self.P[(task, state)] = dur

    def unit(self, unit, task, Bmin=0, Bmax=float('inf'), cost=0, vcost=0,
             tm=0, rmax=0, rinit=0, a=0, b=0, tauinit=0):
        if unit not in self.units:
            self.units.add(unit)
            self.I[unit] = set()
            self.O[unit] = set()
            self.tau[unit] = 0
            self.Rmax[unit] = 0
            self.Rinit[unit] = 0
            self.Rinit0[unit] = 0
            self.a[unit] = 0
            self.b[unit] = 0
            self.deg[unit] = degradationModel(unit)
        if task not in self.tasks:
            self.task(task)
        self.I[unit].add(task)
        self.K[task].add(unit)
        self.Bmin[(task, unit)] = Bmin
        self.Bmax[(task, unit)] = Bmax
        self.cost[(task, unit)] = cost
        self.vcost[(task, unit)] = vcost
        self.tau[unit] = max(tm, self.tau[unit])  # TODO: max is dumb
        self.Rmax[unit] = max(rmax, self.Rmax[unit])
        self.Rinit[unit] = max(rinit, self.Rinit[unit])
        self.Rinit0[unit] = max(rinit, self.Rinit[unit])
        self.a[unit] = max(a, self.a[unit])
        self.b[unit] = max(b, self.b[unit])
        self.tauinit[unit] = tauinit

    def opmode(self, opmode):
        if opmode not in self.opmodes:
            self.opmodes.add(opmode)
            self.modeorder[opmode] = max([-1] + [i for i in
                                                 self.modeorder.values()]) + 1

    def ijkdata(self, task, unit, opmode,
                dur=1, wear=0, sd=0, pinit=0, Binit=0):
        if opmode not in self.O[unit]:
            self.O[unit].add(opmode)
        self.p[task, unit, opmode] = dur
        self.D[task, unit, opmode] = wear
        self.pinit[task, unit, opmode] = pinit
        self.Binit[task, unit, opmode] = Binit
        self.deg[unit].set_op_mode(task + "-" + opmode, wear/dur,
                                   sd/np.sqrt(dur))

    def changeover(self, task1, task2, dur):
        self.changeoverTime[(task1, task2)] = dur

    def pprint(self):
        for task in sorted(self.tasks):
            print('\nTask:', task)
            print('    S[{0:s}]:'.format(task), self.S[task])
            print('    S_[{0:s}]:'.format(task), self.S_[task])
            print('    K[{0:s}]:'.format(task), self.K[task])
            print('    p[{0:s}]:'.format(task), self.p[task])

        for state in sorted(self.states):
            print('\nState:', state)
            print('    T[{0:s}]:'.format(state), self.T[state])
            print('    T_[{0:s}]:'.format(state), self.T_[state])
            print('    C[{0:s}]:'.format(state), self.C[state])
            print('    init[{0:s}]:'.format(state), self.init[state])

        for unit in sorted(self.units):
            print('\nUnit:', unit)
            print('    I[{0:s}]:'.format(unit), self.I[unit])

        print('\nState -> Task Arcs')
        for (task, state) in sorted(self.rho.keys()):
            print('    {0:s} -> {1:s}:'.format(state, task))
            print('        rho:', self.rho[(task, state)])

        print('\nTask -> State Arcs')
        for (task, state) in sorted(self.rho_.keys()):
            print('    {0:s} -> {1:s}:'.format(task, state))
            print('        rho_:', self.rho_[(task, state)])
            print('           P:', self.P[(task, state)])
