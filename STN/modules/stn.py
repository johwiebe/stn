#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Deterministic model of STN with degradation. Based on Biondi et al 2017.
'''
import pyomo.environ as pyomo
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dill
import sys
import csv
from blocks import (blockScheduling, blockSchedulingRobust,
                    blockPlanning, blockPlanningRobust)


class stnModel(object):
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

    def demand(self, state, time, Demand):
        """Add demand to model."""
        self.Demand[state, time] = Demand

    def uncertainty(self, eps):
        self.stn.eps = eps

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
        m.Obj = pyomo.Objective(expr=m.sb.Cost
                                + m.pb.Cost
                                + m.TotSlack*10000,
                                sense=pyomo.minimize)

    def add_blocks(self, TIMEs, TIMEp, **kwargs):
        stn = self.stn
        m = self.model
        m.sb = pyomo.Block()
        self.sb = blockScheduling(stn, TIMEs,
                                  self.Demand, **kwargs)
        self.sb.define_block(m.sb, **kwargs)
        m.pb = pyomo.Block()
        self.pb = blockPlanning(stn, TIMEp,
                                self.Demand, **kwargs)
        self.pb.define_block(m.pb, **kwargs)

    def transfer_next_period(self, **kwargs):
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

    def build(self, T_list, objective="biondi", period=None, **kwargs):
        """Build STN model."""
        assert period is not None
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

        # scheduling and planning block
        Ts = T_list[0]
        dTs = T_list[1]
        Tp = T_list[2]
        dTp = T_list[3]
        Ts_start = period * Ts
        Tp_start = (period + 1) * Ts
        Ts = Ts_start + Ts
        Tp = Ts_start + Tp
        self.add_blocks([Ts_start, Ts, dTs], [Tp_start, Tp, dTp], **kwargs)

        # add continuity constraints to model
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints(**kwargs)

        # add objective function to model
        self.add_objective()

    def solve(self, T_list, periods=1, solver='cplex', prefix='',
              rdir='results', solverparams=None,
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
        self.solver = pyomo.SolverFactory(solver)
        if solverparams is not None:
            for key, value in solverparams.items():
                self.solver.options[key] = value
        prefix_old = prefix

        # Rolling horizon
        for period in range(0, periods):
            if periods > 1:
                prefix = prefix_old + "_" + str(period)
            # Build model
            self.build(T_list, period=period, **kwargs)
            logfile = rdir + "/" + prefix + "STN.log"
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
                    with open(rdir+"/"+prefix+'output.txt', 'w') as f:
                        f.write("STN Output:")
                        self.model.display(ostream=f)
                    with open(rdir+"/"+prefix+'STN.pyomo', 'wb') as dill_file:
                        dill.dump(self.model, dill_file)
                if gantt:
                    self.sb.gantt(prefix=prefix, rdir=rdir)
                    self.pb.gantt(prefix=prefix, rdir=rdir)
                if trace:
                    self.sb.trace(prefix=prefix, rdir=rdir)
                    self.pb.trace(prefix=prefix, rdir=rdir)
                if periods > 1:
                    self.transfer_next_period(**kwargs)
                # Add current model to list
                self.m_list.append(self.model)
            else:
                break

    def loadres(self, f="STN.pyomo"):
        with open(f, 'rb') as dill_file:
            self.model = dill.load(dill_file)

    def resolve(self, solver='cplex', prefix=''):  # FIX: self -> stn
        for j in self.units:
            for t in self.sb.TIME:
                for i in self.I[j]:
                    for k in self.O[j]:
                        self.model.sb.W[i, j, k, t].fixed = True
                        self.model.sb.B[i, j, k, t].fixed = True
                self.model.sb.M[j, t].fixed = True
            for t in self.pb.TIME:
                for i in self.I[j]:
                    for k in self.O[j]:
                        self.model.pb.N[i, j, k, t].fixed = True
                        self.model.pb.A[i, j, t].fixed = True
                        # self.model.pb.Mode[j,k,t].fixed = True
                self.model.pb.M[j, t].fixed = True
        self.model.preprocess()
        self.solver = pyomo.SolverFactory(solver)
        # results = self.solver.solve(self.model,tee=True,
        #                             logfile="results/r"+prefix+"STN.log")
        self.solver.options['dettimelimit'] = 500000
        self.solver.solve(self.model,
                          tee=True,
                          logfile="results/r"+prefix+"STN.log").write()
        with open("results/r"+prefix+'output.txt', 'w') as f:
            f.write("STN Output:")
            self.model.display(ostream=f)
        with open("results/r"+prefix+'STN.pyomo', 'wb') as dill_file:
            dill.dump(self.model, dill_file)

    def reevaluate(self, prefix):  # FIX: self -> stn
        m = self.model

        # Recalculate F and R
        constraintCheck = True
        for j in self.units:
            rhs = self.Rinit[j]
            for t in self.sb.TIME:
                # residual life balance
                if m.sb.M[j, t]():
                    m.sb.F[j, t] = self.Rmax[j] - rhs
                    rhs = self.Rmax[j]
                for i in self.I[j]:
                    for k in self.O[j]:
                        rhs -= self.D[i, j, k]*m.sb.W[i, j, k, t]()
                m.sb.R[j, t] = max(rhs, 0)
                if rhs < 0:
                    constraintCheck = False
            for t in self.pb.TIME:
                # residual life balance
                for i in self.I[j]:
                    for k in self.O[j]:
                        rhs -= self.D[i, j, k]*m.pb.N[i, j, k, t]()
                if m.pb.M[j, t]():
                    m.pb.F[j, t] = self.Rmax[j] - rhs
                    rhs = self.Rmax[j]
                m.pb.R[j, t] = max(rhs, 0)
                if rhs < 0:
                    constraintCheck = False
        # Recalculate cost
        costStorage = 0
        for s in self.states:
            costStorage += (self.scost[s]
                            * (m.sb.Sfin[s]()
                               + sum([m.pb.S[s, t]() for t in self.pb.TIME])))

        costMaintenance = 0
        costWear = 0
        for j in self.units:
            for t in self.sb.TIME:
                costMaintenance += (self.a[j]*m.sb.M[j, t]()
                                    - self.b[j]*m.sb.F[j, t]()/self.Rmax[j])
                for i in self.I[j]:
                    for k in self.O[j]:
                        costWear += self.D[i, j, k]*m.sb.W[i, j, k, t]()
            for t in self.pb.TIME:
                costMaintenance += (self.a[j]*m.pb.M[j, t]()
                                    - self.b[j]*m.pb.F[j, t]()/self.Rmax[j])
                for i in self.I[j]:
                    for k in self.O[j]:
                        costWear += self.D[i, j, k]*m.pb.N[i, j, k, t]()
        m.CostStorage = costStorage
        m.CostMaintenance = costMaintenance
        m.CostWear = costWear
        return constraintCheck

    def getD(self):
        return self.D

    def get_gap(self):
        return self.gapmax, self.gapmean, self.gapmin

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

    def get_unit_profile(self, j, full=True):
        cols = ["period", "time", "unit", "task", "mode"]
        prods = set([p[0] for p in self.Demand.keys()])
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

    def get_production_targets(self):
        m = self.model
        stn = self.stn
        cols = ["time"]
        prods = set([p[0] for p in self.Demand.keys()])
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


class stnModelRobust(stnModel):
    def __init__(self, stn=None):
        super().__init__(stn)

    def build(self, T_list, objective="biondi", period=None, tindexed=True,
              **kwargs):
        assert period is not None
        super().build(T_list, objective=objective, period=period,
                      tindexed=tindexed, **kwargs)

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
                                        **kwargs)
        self.sb.define_block(m.sb, decisionrule=decisionrule, **kwargs)
        self.pb = blockPlanningRobust(stn,
                                      np.array([t for t in TIMEp]),
                                      self.Demand,
                                      decisionrule=decisionrule,
                                      **kwargs)
        self.pb.define_block(m.pb, decisionrule=decisionrule, **kwargs)

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

        # constants
        self.U = 100                # big U
        self.eps = 0.0              # Maximum deviation for uncertain D's FIX!
        self.alpha = 0.5            # uncertainty set size parameter

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

        # characterization of units indexed by (task, unit, operating mode)
        self.p = {}                 # task duration
        self.D = {}                 # wear coefficient
        self.pinit = {}             # initial processing time left
        self.Binit = {}             # initial amount being processed

        # dictionaries indexed by (task,task)
        self.changeoverTime = {}    # time required for task1 -> task2

    # defines states as .state(name, capacity, init)
    def state(self, name, capacity=float('inf'), init=0, price=0, scost=0):
        self.states.add(name)       # add to the set of states
        self.C[name] = capacity     # state capacity
        self.init[name] = init      # state initial value
        self.T[name] = set()        # tasks which feed this state (inputs)
        self.T_[name] = set()       # tasks fed from this state (outputs)
        self.price[name] = price    # per unit price of each state
        self.scost[name] = scost    # storage cost per (planning) interval

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
#        self.p[task] = max(self.p[task],dur)

    def unit(self, unit, task, Bmin=0, Bmax=float('inf'), cost=0, vcost=0,
             tm=0, rmax=0, rinit=0, a=0, b=0, tauinit=0):
        if unit not in self.units:
            self.units.add(unit)
            self.I[unit] = set()
            self.O[unit] = set()
            self.tau[unit] = 0
            self.Rmax[unit] = 0
            self.Rinit[unit] = 0
            self.a[unit] = 0
            self.b[unit] = 0
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
        self.a[unit] = max(a, self.a[unit])
        self.b[unit] = max(b, self.b[unit])
        self.tauinit[unit] = tauinit

    def opmode(self, opmode):
        if opmode not in self.opmodes:
            self.opmodes.add(opmode)

    def ijkdata(self, task, unit, opmode,
                dur=1, wear=0, pinit=0, Binit=0, sd=0):
        if opmode not in self.O[unit]:
            self.O[unit].add(opmode)
        self.p[task, unit, opmode] = dur
        self.D[task, unit, opmode] = wear
        self.pinit[task, unit, opmode] = pinit
        self.Binit[task, unit, opmode] = Binit

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
