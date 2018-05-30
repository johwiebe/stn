#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheduling horizon of STN model with degradation.

This module implements the scheduling horizon of the STN model with degradation
by Biondi et al 2017.

The implementation of the STN is based on the implementation by Jeffrey Kantor
2017 (https://github.com/jckantor/STN-Scheduler)

"""

import pyomo.environ as pyomo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import seaborn as sns
import dill
import time
import sys
import matplotlib.pyplot as plt
from deg import calc_p_fail


class stnBlock(object):
    """Template for planning and scheduling blocks."""

    def __init__(self, stn, T_list, Demand={}, prfx="", **kwargs):
        """
        stn: object of class stnStruct
        T_list: [Tstart, Tend, dT]
        Demand: dictionary of demands indexed by (product, time)
        """
        self.stn = stn
        self.T = T_list[1]
        self.dT = T_list[2]
        TIME = range(T_list[0], self.T, self.dT)
        self.TIME = np.array(TIME)
        self.Demand = Demand
        self.alpha = 0.5
        self.rid = 0
        self.prefix = ''
        self.rdir = 'results'
        self.ttot = 0
        self.prfx = prfx
        self.inf = False

    def demand(self, state, time, Demand):
        """Set demand for state at time."""
        self.Demand[state, time] = Demand

    def build(self, alpha=0.5, rdir='results', prefix='', **kwargs):
        """
        Initializes and builds model. Only call this if block is
        used individually.
        """
        self.rdir = rdir
        self.prefix = prefix
        try:
            df = pd.read_pickle(self.rdir+"/"+self.prefix+"results.pkl")
            self.rid = max(df["id"]) + 1
        except IOError:
            pass
        self.prfx = self.rdir + "/" + self.prefix + str(self.rid)
        self.model = pyomo.ConcreteModel()
        m = self.model
        stn = self.stn
        if alpha != self.alpha:
            for j in stn.units:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        tm = i + "-" + k
                        p = stn.p[i, j, k]
                        D = stn.deg[j].get_mu(tm, p)
                        eps = stn.deg[j].get_eps(alpha, tm, p)
                        stn.D[i, j, k] = D*(1 + eps)
            self.alpha = alpha
        self.define_block(m, **kwargs)
        # Add objective to model
        m.Obj = pyomo.Objective(expr=m.Cost,
                                sense=pyomo.minimize)

    def solve(self, solver='cplex', solverparams=None,
              save=False, trace=False, gantt=True, **kwargs):
        """Solves the block. Only call when block is used individually."""
        ts = time.time()
        self.solver = pyomo.SolverFactory(solver)
        if solverparams is not None:
            for key, value in solverparams.items():
                self.solver.options[key] = value

        logfile = self.prfx + "STN.log"
        results = self.solver.solve(self.model,
                                    tee=True,
                                    # keepfiles=True,
                                    # symbolic_solver_labels=True,
                                    logfile=logfile)
        results.write()
        if ((results.solver.status == SolverStatus.ok) and
            (results.solver.termination_condition ==
             TerminationCondition.optimal or
             results.solver.termination_condition ==
             TerminationCondition.maxTimeLimit)):
            if save:
                with open(self.prfx+'output.txt', 'w') as f:
                    f.write("STN Output:")
                    self.model.display(ostream=f)
                with open(self.prfx+'STN.pyomo', 'wb') as dill_file:
                    dill.dump(self.model, dill_file)
            if gantt:
                self.gantt()
            if trace:
                self.trace()
        self.ttot = time.time() - ts
        if (results.solver.termination_condition ==
                TerminationCondition.infeasible):
            self.inf = True
        else:
            self.inf = False

    def loadres(self, f="STN.pyomo"):
        with open(f, 'rb') as dill_file:
            self.model = dill.load(dill_file)
            self.b = self.model

    def define_block(self, b, **kwargs):
        """Add variables, constraints, and objective function to model."""
        self.b = b
        b.TIME = self.TIME
        b.T = self.T
        b.dT = self.dT
        b.cons = pyomo.ConstraintList()

        # add variables, constraints, and objective to model
        self.add_vars(**kwargs)
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints(**kwargs)
        self.add_objective(**kwargs)

    def add_vars(self, **kwargs):
        raise NotImplementedError

    def add_unit_constraints(self):
        raise NotImplementedError

    def add_state_constraints(self):
        raise NotImplementedError

    def add_deg_constraints(self, **kwargs):
        raise NotImplementedError

    def add_objective(self, **kwargs):
        raise NotImplementedError

    def trace(self):
        raise NotImplementedError

    def gantt(self):
        raise NotImplementedError


class blockScheduling(stnBlock):
    """Implements deterministic constraints for the scheduling horizon"""

    def __init__(self, stn, TIME, Demand={}, **kwargs):
        super().__init__(stn, TIME, Demand, **kwargs)

    def add_unit_constraints(self):
        """Add unit constraints to block"""
        b = self.b
        stn = self.stn
        for j in stn.units:
            for t in self.TIME:
                rhs = 1
                lhs = 0
                # check if task is still running on unit j
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if t - self.TIME[0] < stn.pinit[i, j, k]:
                            rhs = 0
                        for tprime in self.TIME[(self.TIME <= t)
                                                & (self.TIME
                                                   >= t
                                                   - stn.p[i, j, k]
                                                   + self.dT)]:
                            lhs += b.W[i, j, k, tprime]
                # check if maintenance is going on on unit j
                if t - self.TIME[0] < stn.tauinit[j]:
                    rhs = 0
                for tprime in self.TIME[(self.TIME <= t)
                                        & (self.TIME
                                           >= t
                                           - stn.tau[j]
                                           + self.dT)]:
                    lhs += b.M[j, tprime]
                # a unit can only be allocated to one task
                b.cons.add(lhs <= rhs)

                # capacity constraints (see Konkili, Sec. 3.1.2)
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        b.cons.add(b.W[i, j, k, t]*stn.Bmin[i, j]
                                   <= b.B[i, j, k, t])
                        b.cons.add(b.B[i, j, k, t]
                                   <= b.W[i, j, k, t]*stn.Bmax[i, j])

    def add_state_constraints(self):
        """Add state constraints to block"""
        b = self.b
        stn = self.stn
        for s in stn.states:
            rhs = stn.init[s]
            for t in self.TIME:
                # state capacity constraint
                b.cons.add(b.S[s, t] - b.Sslack[s, t] <= stn.C[s])
                # state mass balanace
                for i in stn.T_[s]:
                    for j in stn.K[i]:
                        for k in stn.O[j]:
                            if (t >= stn.P[(i, s)]
                                    + stn.p[i, j, k]
                                    + self.TIME[0]):
                                tprime = max(self.TIME[self.TIME
                                                       <= t
                                                       - stn.p[i, j, k]
                                                       - stn.P[(i, s)]])
                                rhs += stn.rho_[(i, s)]*b.B[i, j, k,
                                                            tprime]
                            if t - self.TIME[0] == stn.pinit[i, j, k]:
                                rhs += stn.rho_[(i, s)]*stn.Binit[i, j, k]
                for i in stn.T[s]:
                    for j in stn.K[i]:
                        for k in stn.O[j]:
                            rhs -= stn.rho[(i, s)]*b.B[i, j, k, t]
                # if (s, t - self.dT) in self.Demand:
                #     rhs -= self.Demand[s, t - self.dT]
                b.cons.add(b.S[s, t] == rhs)
                rhs = b.S[s, t]

    def add_deg_constraints(self, **kwargs):
        """Add residual life constraints to block"""
        b = self.b
        stn = self.stn
        for j in stn.units:
            rhs = stn.Rinit[j]
            for t in self.TIME:
                # constraints on F[j,t] and R[j,t]
                # TODO: these are obsolete?
                b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])
                b.cons.add(b.F[j, t] <= stn.Rmax[j] - rhs)
                b.cons.add(b.F[j, t] >= stn.Rmax[j]*b.M[j, t] - rhs)
                b.cons.add(0 <= b.R[j, t])
                b.cons.add(b.R[j, t] <= stn.Rmax[j]*(1 - b.M[j, t]))
                # residual life balance
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        rhs += stn.D[i, j, k]*b.W[i, j, k, t]
                # rhs += b.F[j, t]
                # b.cons.add(b.R[j, t] == rhs)
                b.cons.add(b.R[j, t] >= rhs - stn.Rmax[j]*b.M[j, t])
                b.cons.add(b.R[j, t] <= rhs)
                rhs = b.R[j, t]

    def add_objective(self, objective=None, **kwargs):
        """Add objective function to model."""
        assert objective is not None
        b = self.b
        stn = self.stn

        costStorage = 0
        for s in stn.states:
            costStorage += stn.scost[s]*(b.Sfin[s])
        b.cons.add(b.CostStorage == costStorage)

        costMaintenance = 0
        costMaintenanceEarly = 0
        for j in stn.units:
            for t in self.TIME:
                costMaintenance += ((stn.a[j] - stn.b[j])
                                    * b.M[j, t])
                if objective == "biondi":
                    costMaintenanceEarly += (stn.b[j]
                                             * (b.M[j, t]
                                                - b.F[j, t]
                                                / stn.Rmax[j]))
        b.cons.add(b.CostMaintenance == costMaintenance)
        b.cons.add(b.CostMaintenanceEarly == costMaintenanceEarly)
        costWear = 0
        if objective == "biondi":
            costWear += self.calc_cost_wear()
            b.cons.add(b.Cost == b.CostStorage + b.CostMaintenance +
                       + b.CostWear + b.CostMaintenanceEarly)
        elif objective == "terminal":
            b.cons.add(b.Cost == b.CostStorage + b.CostMaintenance +
                       b.CostMaintenanceFinal)
        else:
            raise KeyError("KeyError: unknown objective %s" % objective)
        b.cons.add(b.CostWear == costWear)

    def calc_cost_wear(self):
        """Calculate wear penalty (Biondi)."""
        b = self.b
        stn = self.stn

        costTotal = 0
        for t in self.TIME:
            costWear = 0
            for j in stn.units:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        costWear += stn.D[i, j, k]*b.W[i, j, k, t]
            costTotal += costWear
        return costTotal

    def add_demand_constraint(self):
        """Add demand constraint when block is used individually."""
        stn = self.stn
        b = self.b
        for s in stn.states:
            rhs = b.S[s, self.T - self.dT]
            for i in stn.T_[s]:
                for j in stn.K[i]:
                    for k in stn.O[j]:
                        rhs += (stn.rho_[(i, s)]
                                * b.B[i, j, k,
                                      self.T
                                      - stn.p[i, j, k]])
            # Subtract demand from last scheduling period
            if s in self.Demand:
                rhs -= self.Demand[s]
            b.cons.add(b.Sfin[s] == rhs)
            b.cons.add(0 <= b.Sfin[s] <= stn.C[s])

    def build(self, objective="terminal", **kwargs):
        """Only use when block is solved individually."""
        stn = self.stn
        super().build(objective=objective, **kwargs)
        b = self.b
        # calculate terminal maint cost
        self.calc_cost_maintenance_terminal()
        # add demand constraint
        self.add_demand_constraint()
        # add slack for unfullfilled demand
        for s in stn.states:
            for t in self.TIME:
                b.cons.add(b.Sslack[s, t] == 0)

    def define_block(self, b, **kwargs):
        super().define_block(b, **kwargs)

    def add_vars(self, **kwargs):
        """Add variables to model."""
        b = self.b
        stn = self.stn

        # W[i,j,k,t] 1 if task i starts in unit j and operating mode
        # k at time t
        b.W = pyomo.Var(stn.tasks, stn.units, stn.opmodes, self.TIME,
                        domain=pyomo.Binary)

        # M[j,t] 1 if unit j undergoes maintenance at time t
        b.M = pyomo.Var(stn.units, self.TIME, domain=pyomo.Binary)

        # R[j,t] residual life time of unit j at time t
        b.R = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        # F[j,t] residual life restoration during maintenance
        b.F = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        # B[i,j,k,t] size of batch assigned to task i in unit j at time t
        b.B = pyomo.Var(stn.tasks, stn.units, stn.opmodes, self.TIME,
                        domain=pyomo.NonNegativeReals)

        # S[s,t] inventory of state s at time t
        b.S = pyomo.Var(stn.states, self.TIME, domain=pyomo.NonNegativeReals)

        # Q[j,t] inventory of unit j at time t
        b.Q = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        # Variables for continuity between scheduling and planning horizon
        b.Sfin = pyomo.Var(stn.states, domain=pyomo.NonNegativeReals)

        # cost components and total cost
        b.CostStorage = pyomo.Var(domain=pyomo.NonNegativeReals)
        b.CostMaintenance = pyomo.Var(domain=pyomo.NonNegativeReals)
        b.CostMaintenanceFinal = pyomo.Var(domain=pyomo.NonNegativeReals)
        b.CostMaintenanceEarly = pyomo.Var(domain=pyomo.NonNegativeReals)
        b.CostWear = pyomo.Var(domain=pyomo.NonNegativeReals)
        b.Cost = pyomo.Var(domain=pyomo.NonNegativeReals)
        # slack for unfulfilled demand
        b.Sslack = pyomo.Var(stn.states, self.TIME,
                             domain=pyomo.NonNegativeReals)

    def calc_cost_maintenance_terminal(self):
        """Calculate terminal cost of maintenance."""
        stn = self.stn
        b = self.b
        costFinal = 0
        for j in stn.units:
            costFinal += ((b.R[j, self.T - self.dT]
                           / stn.Rmax[j])
                          * (stn.a[j] - stn.b[j]))
        b.cons.add(b.CostMaintenanceFinal == costFinal)

    def get_cost_maint_terminal(self, b=None):
        """Calculate terminal cost of maintenance."""
        stn = self.stn
        if b is None:
            b = self.b
        costFinal = 0
        for j in stn.units:
            costFinal += ((b.R[j, b.T - b.dT]()
                           / stn.Rmax[j])
                          * (stn.a[j] - stn.b[j]))
        return costFinal

    def check_for_task(self, j, t):
        """Check if task is being performed on unit j at time t."""
        b = self.b
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

    def get_unit_profile(self, units=None, full=False, save=True):
        """Get sequence of operating modes for unit j."""
        cols = ["time", "unit", "task", "mode"]
        prods = set(self.Demand.keys())
        demand = []
        for p in prods:
            cols.append(p)
            demand.append(0)
        profile = pd.DataFrame(columns=cols)
        if units is None:
            units = self.stn.units
        elif type(units) == str:
            units = set([units])
        for j in units:
            tend = 0
            for t in self.TIME:
                tend_old = tend
                if t >= tend:
                    [i, k, tend] = self.check_for_task(j, t)
                line = [t, j, i, k]
                for n, p in enumerate(prods):
                    if (p) in self.Demand:
                        demand[n] = self.Demand[p]
                line += demand
                if full or t >= tend_old:
                    profile = profile.append(pd.Series(line,
                                                       index=cols),
                                             ignore_index=True)
        profile["id"] = self.rid
        if save:
            try:
                df2 = pd.read_pickle(self.rdir+"/"+self.prefix+"profile.pkl")
                df2 = df2.append(profile)
            except IOError:
                df2 = profile
            df2.to_pickle(self.rdir+"/"+self.prefix+"profile.pkl")
            df2.to_csv(self.rdir+"/"+self.prefix+"profile.csv")
        return profile

    def gantt(self):
        """
        Generate gantt graph.
            prefix: prepended to file name
            rdir: directory to store file
        """
        b = self.b
        stn = self.stn

        gap = (self.T - self.TIME[0])/400
        idx = 1
        lbls = []
        ticks = []

        # create a list of units sorted by time of first assignment
        jstart = {j: self.T+1 for j in stn.units}
        for j in stn.units:
            for i in stn.I[j]:
                for k in stn.O[j]:
                    for t in self.TIME:
                        # print(self.model.W[i,j,k,t]())
                        if b.W[i, j, k, t]() > 0:
                            jstart[j] = min(jstart[j], t)
        jsorted = [j for (j, t) in sorted(jstart.items(),  key=lambda x: x[1])]
        jsorted = sorted(stn.units)

        mode_palette = sns.color_palette("YlOrRd", len(stn.opmodes))
        mode_col = {}
        for mode in stn.opmodes:
            mode_col.update({mode: mode_palette[stn.modeorder[mode]]})

        # number of horizontal bars to draw
        nbars = -1
        for j in jsorted:
            for i in sorted(stn.I[j]):
                nbars += 1
            nbars += 0.5
        plt.figure(figsize=(12, (nbars+1)/2))

        for j in jsorted:
            idx -= 0.5
            idx0 = idx
            for t in self.TIME:
                idx = idx0
                for i in sorted(stn.I[j]):
                    idx -= 1
                    if t == self.TIME[0]:
                        ticks.append(idx)
                        plt.plot([self.TIME[0], self.T],
                                 [idx, idx], lw=24,
                                 alpha=.3, color='y')
                        lbls.append("{0:s} -> {1:s}".format(j, i))
                        for k in stn.O[j]:
                            if stn.pinit[i, j, k] > self.dT/2:
                                plt.plot([t, t+stn.pinit[i, j, k]],
                                         [idx, idx], 'k',  lw=24,
                                         alpha=0.5, solid_capstyle='butt')
                                plt.plot([t+gap, t+stn.pinit[i, j, k]-gap],
                                         [idx, idx], color=mode_col[k], lw=20,
                                         solid_capstyle='butt')
                                txt = "{0:.2f}".format(stn.Binit[i, j, k])
                                plt.text(t+stn.pinit[i, j, k]/2, idx,  txt,
                                         weight='bold', ha='center',
                                         va='center')
                    for k in stn.O[j]:
                        if b.W[i, j, k, t]() > 0.5:
                            plt.plot([t, t+stn.p[i, j, k]],
                                     [idx, idx], 'k',  lw=24,
                                     alpha=0.5, solid_capstyle='butt')
                            plt.plot([t+gap, t+stn.p[i, j, k]-gap],
                                     [idx, idx], color=mode_col[k], lw=20,
                                     solid_capstyle='butt')
                            txt = "{0:.2f}".format(b.B[i, j, k, t]())
                            plt.text(t+stn.p[i, j, k]/2, idx,  txt,
                                     weight='bold', ha='center', va='center')
                    if b.M[j, t]() > 0.5:
                        plt.plot([t, t+stn.tau[j]],
                                 [idx, idx], 'k',  lw=24,
                                 alpha=0.5, solid_capstyle='butt')
                        plt.plot([t+gap, t+stn.tau[j]-gap],
                                 [idx, idx], color="grey", lw=20,
                                 solid_capstyle='butt')
                        plt.text(t+stn.tau[j]/2, idx, "Maintenance",
                                 weight='bold', ha='center', va='center')

        plt.xlim(self.TIME[0], self.T)
        plt.ylim(-nbars-0.5, 0)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls)
        plt.savefig(self.prfx+'gantt_scheduling.png')

        plt.close("all")

    def trace(self):
        """
        Generate trace.
            prefix: prepended to file name
            rdir: directory to store file
        """
        # abbreviations
        b = self.b
        stn = self.stn

        oldstdout = sys.stdout
        sys.stdout = open(self.prfx + 'trace_scheduling.txt', 'w')
        print("\nStarting Conditions")
        print("\n    Initial State Inventories are:")
        for s in stn.states:
            print("        {0:10s}  {1:6.1f} kg".format(s, stn.init[s]))

        # for tracking unit assignments
        # t2go[j]['assignment'] contains the task to which unit j is currently
        # assigned
        # t2go[j]['t'] is the time to go on equipment j
        time2go = {j: {'assignment': 'None', 't': 0} for j in stn.units}

        for t in b.TIME:
            print("\nTime =", t, "hr")

            # create list of instructions
            strList = []

            # first unload units
            for j in stn.units:
                time2go[j]['t'] -= b.dT
                fmt = 'Transfer {0:.2f} kg from {1:s} to {2:s}'
                for i in stn.I[j]:
                    for s in stn.S_[i]:
                        for k in stn.O[j]:
                            ts = t-stn.p[i, j, k]
                            if ts >= b.TIME[0]:
                                tend = max(b.TIME[b.TIME <= ts])
                                amt = (stn.rho_[(i, s)]
                                       * b.B[i, j, k, tend]())
                                if amt > 0:
                                    strList.append(fmt.format(amt, j, s))

            for j in stn.units:
                # release units from tasks
                fmt = 'Release {0:s} from {1:s}'
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if t-stn.p[i, j, k] >= b.TIME[0]:
                            tend = max(b.TIME[b.TIME
                                              <= t
                                              - stn.p[i, j, k]])
                            if b.W[i, j, k, tend]() > 0:
                                strList.append(fmt.format(j, i))
                                time2go[j]['assignment'] = 'None'
                                time2go[j]['t'] = 0

                # assign units to tasks
                fmt = ('Assign {0:s} to {1:s} for {2:.2f} kg batch for {3:.1f}'
                       'hours (Mode: {4:s})')
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        amt = b.B[i, j, k, t]()
                        if b.W[i, j, k, t]() > 0.5:
                            strList.append(fmt.format(j, i, amt,
                                                      stn.p[i, j, k], k))
                            time2go[j]['assignment'] = i
                            time2go[j]['t'] = stn.p[i, j, k]

                # transfer from states to tasks/units
                fmt = 'Transfer {0:.2f} from {1:s} to {2:s}'
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for s in stn.S[i]:
                            amt = stn.rho[(i, s)] * b.B[i, j, k, t]()
                            if amt > 0:
                                strList.append(fmt.format(amt, s, j))

                # Check if maintenance is done on unit
                fmt = 'Doing maintenance on {0:s}'
                if b.M[j, t]() > 0.5:
                    strList.append(fmt.format(j))

            if len(strList) > 0:
                print()
                idx = 0
                for str in strList:
                    idx += 1
                    print('   {0:2d}. {1:s}'.format(idx, str))

            print("\n    State Inventories are now:")
            for s in stn.states:
                print("        {0:10s}  {1:6.1f} kg".format(s,
                                                            b.S[s, t]()))

        sys.stdout = oldstdout

    def eval(self, save=True):
        cols = ["id", "CostStorage", "CostMaintenance",
                "CostMainenanceFinal", "obj", "inf", "ttot", "gap"]
        cols += self.stn.products
        if not self.inf:
            gap = self.solver._gap
            if gap is None:
                gap = 0
            else:
                gap = self.solver._gap/self.b.Obj()*100
            cost_storage = self.b.CostStorage()
            cost_maint = self.b.CostMaintenance()
            cost_maint_final = self.b.CostMaintenanceFinal()
            obj = self.b.Obj()
            df = pd.DataFrame([[self.rid, cost_storage,
                                cost_maint, cost_maint_final, obj, self.inf,
                                self.ttot, gap] + [self.Demand[p] for p in
                                                   self.stn.products]],
                              columns=cols)
        else:
            df = pd.DataFrame([[self.rid, 0, 0, 0, "Inf", self.inf,
                                self.ttot, "NA"] + [self.Demand[p] for p in
                                                    self.stn.products]],
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


class blockSchedulingRobust(blockScheduling):
    """Implements robust constraints for the scheduling horizon"""

    def __init__(self, stn, TIME, Demand={}, **kwargs):
        super().__init__(stn, TIME, Demand, **kwargs)

    def build(self, decisionrule="continuous", tindexed=False,
              **kwargs):
        super().build(decisionrule=decisionrule, tindexed=tindexed, **kwargs)

    def calc_nominal_R(self, tindexed=None, **kwargs):
        """ calculate R using nominal values of D."""
        assert tindexed is not None
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                rhs = b.R0[j, t]
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if tindexed:
                            for tprime in self.TIME[self.TIME <= t]:
                                rhs += stn.D[i, j, k]*b.Rc[j, t, i, k, tprime]
                        else:
                            rhs += stn.D[i, j, k]*b.Rc[j, t, i, k]
                b.cons.add(b.R[j, t] == rhs)

    def calc_max_R(self, tindexed=None, **kwargs):
        """Calculate R using maximal values of D."""
        assert tindexed is not None
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                rhs = b.R0[j, t]
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if tindexed:
                            for tprime in self.TIME[self.TIME <= t]:
                                rhs += (stn.D[i, j, k]
                                        * (1 + stn.eps[i, j, k])
                                        * b.Rc[j, t, i, k, tprime])
                        else:
                            rhs += (stn.D[i, j, k]
                                    * (1 + stn.eps[i, j, k])
                                    * b.Rc[j, t, i, k])
                b.cons.add(b.Rmax[j, t] == rhs)

    def add_vars_tindexed(self, domain):
        b = self.b
        stn = self.stn

        # pyomo.Variables for residual life affine decision rule
        b.Rc = pyomo.Var(stn.units, self.TIME, stn.tasks, stn.opmodes,
                         self.TIME, domain=domain)

        b.R0 = pyomo.Var(stn.units, self.TIME,
                         domain=pyomo.NonNegativeReals)

        # Dual variables for residual life constraints
        b.ld = pyomo.Var([1, 2, 3, 4], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)
        b.ud = pyomo.Var([1, 2, 3, 4], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)

    # def calc_cost_maintenance_terminal(self):
    #     """Calculate terminal cost of maintenance (robust)."""
    #     b = self.b
    #     stn = self.stn

    #     costFinal = 0
    #     for j in stn.units:
    #         costFinal += ((b.Rmax[j, self.T - self.dT]
    #                        / stn.Rmax[j])
    #                       * (stn.a[j] - stn.b[j]))
    #     b.cons.add(b.CostMaintenanceFinal == costFinal)

    def add_vars_not_tindexed(self, domain):
        b = self.b
        stn = self.stn

        # pyomo.Variables for residual life affine decision rule
        b.Rc = pyomo.Var(stn.units, self.TIME, stn.tasks, stn.opmodes,
                         domain=domain)

        b.R0 = pyomo.Var(stn.units, self.TIME,
                         domain=pyomo.NonNegativeReals)

        # Dual variables for residual life constraints
        b.ld = pyomo.Var([1, 2, 3, 4], stn.units, self.TIME, stn.tasks,
                         stn.opmodes,
                         domain=pyomo.NonNegativeReals)
        b.ud = pyomo.Var([1, 2, 3, 4], stn.units, self.TIME, stn.tasks,
                         stn.opmodes,
                         domain=pyomo.NonNegativeReals)

    def add_vars(self, tindexed=None, decisionrule=None, **kwargs):
        """Define affine decision rule for residual lifetime."""
        assert decisionrule is not None
        assert tindexed is not None
        b = self.b
        stn = self.stn

        if decisionrule == "continuous":
            domain = pyomo.NonNegativeReals
        elif decisionrule == "integer":
            domain = pyomo.NonNegativeIntegers

        if tindexed:
            self.add_vars_tindexed(domain)
        else:
            self.add_vars_not_tindexed(domain)
        b.Rmax = pyomo.Var(stn.units, self.TIME,
                           domain=pyomo.NonNegativeReals)
        super().add_vars()

    def define_block(self, b, decisionrule=None, **kwargs):
        # Define affine decision rule
        super().define_block(b, decisionrule=decisionrule, **kwargs)
        # calculate nominal and maximal R
        self.calc_nominal_R(**kwargs)
        self.calc_max_R(**kwargs)
        assert decisionrule is not None
        # add extra constraints if integer decision rule is used
        if decisionrule == "integer":
            self.add_int_decision_rule_cons(**kwargs)

    def add_deg_constraints(self, tindexed=None, **kwargs):
        """Add degradation constraints."""
        assert tindexed is not None
        if tindexed:
            self.add_deg_constraints_tindexed()
        else:
            self.add_deg_constraints_not_tindexed()

    def add_deg_constraints_tindexed(self):
        """Add robust degredation constraints (tindexed: D[i,j,k,t]).

        Note:
            The residual lifetime R has a different interpretation in the
            robust implementation. It increases from 0 to Rmax instead of
            decreasing from Rmax to 0

        """
        b = self.b
        stn = self.stn

        for j in stn.units:
            # RcLast = stn.Rinit[j]
            for t in self.TIME:
                # constraints on F[j,t] and R[j,t]
                b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])
                b.cons.add(b.F[j, t] <= stn.Rmax[j] - b.R[j, t])
                b.cons.add(b.F[j, t] >= stn.Rmax[j]*b.M[j, t] - b.R[j, t])
                b.cons.add(b.R[j, t] <= stn.Rmax[j])
                # b.cons.add(stn.Rmax[j]*b.M[j, t] <= b.R[j, t])
                # residual life balance
                # inequality 1
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[1, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[1, j, t, i, k, tprime]))
                            b.cons.add(b.ud[1, j, t, i, k, tprime] -
                                       b.ld[1, j, t, i, k, tprime] >= -
                                       b.Rc[j, t, i, k, tprime])
                rhs = b.R0[j, t]
                b.cons.add(lhs <= rhs)

                # inequality 2
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[2, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[2, j, t, i, k, tprime]))
                            b.cons.add(b.ud[2, j, t, i, k, tprime] -
                                       b.ld[2, j, t, i, k, tprime] >=
                                       b.Rc[j, t, i, k, tprime])
                rhs = -b.R0[j, t] + stn.Rmax[j]*(1 - b.M[j, t])
                # rhs = -b.R0[j,t] + stn.Rmax[j]
                b.cons.add(lhs <= rhs)

                # inequality 3
                lhs = 0
                # in the first time period R = Rinit
                if (t == self.TIME[0]):
                    R0Last = stn.Rinit[j]
                    RcLast = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[3, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[3, j, t, i, k, tprime]))

                            if (t > self.TIME[0]):
                                R0Last = b.R0[j, t-self.dT]
                                RcLast = b.Rc[j, t-self.dT, i, k, tprime]

                            if (tprime == t):
                                b.cons.add(b.ud[3, j, t, i, k, tprime]
                                           - b.ld[3, j, t, i, k, tprime]
                                           >=
                                           - b.Rc[j, t, i, k, tprime]
                                           + b.W[i, j, k, t])
                            else:
                                b.cons.add(b.ud[3, j, t, i, k, tprime]
                                           - b.ld[3, j, t, i, k, tprime]
                                           >= RcLast
                                           - b.Rc[j, t, i, k, tprime])
                rhs = (b.R0[j, t] - R0Last
                       # + stn.Rmax[j])
                       + b.M[j, t]*stn.Rmax[j])
                b.cons.add(lhs <= rhs)

                # inequality 4
                lhs = 0
                # in the first time period R = Rinit
                if (t == self.TIME[0]):
                    R0Last = stn.Rinit[j]
                    RcLast = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[4, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[4, j, t, i, k, tprime]))

                            if (t > self.TIME[0]):
                                R0Last = b.R0[j, t-self.dT]
                                RcLast = b.Rc[j, t-self.dT, i, k, tprime]

                            if (tprime == t):
                                b.cons.add(b.ud[4, j, t, i, k, tprime]
                                           - b.ld[4, j, t, i, k, tprime]
                                           >=
                                           + b.Rc[j, t, i, k, tprime]
                                           - b.W[i, j, k, t])
                            else:
                                b.cons.add(b.ud[4, j, t, i, k, tprime]
                                           - b.ld[4, j, t, i, k, tprime]
                                           >= -RcLast
                                           + b.Rc[j, t, i, k, tprime])
                rhs = -b.R0[j, t] + R0Last
                b.cons.add(lhs <= rhs)

    def add_deg_constraints_not_tindexed(self):
        """Add robust degredation constraints (not tindexed: D[i,j,k]).

        Note:
            The residual lifetime R has a different interpretation in the
            robust implementation. It increases from 0 to Rmax instead of
            decreasing from Rmax to 0

        """
        b = self.b
        stn = self.stn

        for j in stn.units:
            # RcLast = stn.Rinit[j]
            for t in self.TIME:
                # constraints on F[j,t] and R[j,t]
                # b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])
                # b.cons.add(b.F[j, t] <= stn.Rmax[j] - b.R[j, t])
                # b.cons.add(b.F[j, t] >= stn.Rmax[j]*b.M[j, t] - b.R[j, t])
                b.cons.add(b.R[j, t] <= stn.Rmax[j])
                # b.cons.add(stn.Rmax[j]*b.M[j, t] <= b.R[j, t])
                # residual life balance
                # inequality 1
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[1, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[1, j, t, i, k]))
                        b.cons.add(b.ud[1, j, t, i, k] -
                                   b.ld[1, j, t, i, k] >= -
                                   b.Rc[j, t, i, k])
                rhs = b.R0[j, t]
                b.cons.add(lhs <= rhs)

                # inequality 2
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[2, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[2, j, t, i, k]))
                        b.cons.add(b.ud[2, j, t, i, k] -
                                   b.ld[2, j, t, i, k] >=
                                   b.Rc[j, t, i, k])
                rhs = -b.R0[j, t] + stn.Rmax[j]*(1 - b.M[j, t])
                # rhs = -b.R0[j,t] + stn.Rmax[j]
                b.cons.add(lhs <= rhs)

                # inequality 3
                lhs = 0
                # in the first time period R = Rinit
                if (t == self.TIME[0]):
                    R0Last = stn.Rinit[j]
                    RcLast = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[3, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[3, j, t, i, k]))

                        if (t > self.TIME[0]):
                            R0Last = b.R0[j, t-self.dT]
                            RcLast = b.Rc[j, t-self.dT, i, k]

                        b.cons.add(b.ud[3, j, t, i, k]
                                   - b.ld[3, j, t, i, k]
                                   >=
                                   RcLast
                                   - b.Rc[j, t, i, k]
                                   + b.W[i, j, k, t])
                rhs = (b.R0[j, t] - R0Last
                       # + stn.Rmax[j])
                       + b.M[j, t]*stn.Rmax[j])
                b.cons.add(lhs <= rhs)

                # inequality 4
                lhs = 0
                # in the first time period R = Rinit
                if (t == self.TIME[0]):
                    R0Last = stn.Rinit[j]
                    RcLast = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[4, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[4, j, t, i, k]))

                        if (t > self.TIME[0]):
                            R0Last = b.R0[j, t-self.dT]
                            RcLast = b.Rc[j, t-self.dT, i, k]

                        b.cons.add(b.ud[4, j, t, i, k]
                                   - b.ld[4, j, t, i, k]
                                   >=
                                   b.Rc[j, t, i, k]
                                   - RcLast
                                   - b.W[i, j, k, t])
                rhs = -b.R0[j, t] + R0Last
                b.cons.add(lhs <= rhs)

    def add_int_decision_rule_cons(self, tindexed=None, **kwargs):
        """Define affine integer decision rule for residual lifetime."""
        assert tindexed is not None
        b = self.b
        stn = self.stn

        # add constraints for decision rule
        if tindexed:
            for t in self.TIME:
                for j in stn.units:
                    for i in stn.I[j]:
                        for k in stn.O[j]:
                            b.cons.add(b.Rc[j, t, i, k, t] == b.W[i, j, k, t])
                            b.cons.add(b.Rc[j, t, i, k, t] <= 1 - b.M[j, t])
                            for tprime in self.TIME[self.TIME < t]:
                                b.cons.add(b.Rc[j, t, i, k, tprime]
                                           >= b.Rc[j, t - self.dT,
                                                   i, k, tprime]
                                           - b.M[j, t])
                                b.cons.add(b.Rc[j, t, i, k, tprime]
                                           <= 1 - b.M[j, t])
                                b.cons.add(b.Rc[j, t, i, k, tprime]
                                           <= b.Rc[j, t - self.dT,
                                                   i, k, tprime])
        else:
            raise NotImplementedError


class blockPlanning(stnBlock):
    """Implements deterministic constraints for the planning horizon"""

    def __init__(self, stn, TIME, Demand={}, **kwargs):
        super().__init__(stn, TIME, Demand, **kwargs)

    def add_unit_constraints(self):
        """Add unit constraints to block."""
        b = self.b
        stn = self.stn
        # unit constraints
        for j in stn.units:
            lhs = b.Ntransfer[j]  # take Ntransfer into account
            for t in self.TIME:
                # a unit can only be allocated to one task
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += b.N[i, j, k, t]*stn.p[i, j, k]
                lhs += b.M[j, t]*stn.tau[j]
                b.cons.add(lhs <= self.dT)

                # capacity constraints (see Konkili, Sec. 3.1.2)
                for i in stn.I[j]:
                    b.cons.add(sum([b.N[i, j, k, t] for k in stn.O[j]])
                               * stn.Bmin[i, j]
                               <= b.A[i, j, t])
                    b.cons.add(b.A[i, j, t]
                               <= sum([b.N[i, j, k, t] for k in stn.O[j]])
                               * stn.Bmax[i, j])

                # operating mode constraints
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        b.cons.add(b.N[i, j, k, t]
                                   # <= stn.U*b.Mode[j, k, t])
                                   <= b.dT/stn.p[i, j, k]*b.Mode[j, k, t])
                b.cons.add(sum([b.Mode[j, k, t] for k in stn.O[j]]) == 1)
                lhs = 0  # set lhs to zero for next time period

    def add_state_constraints(self):
        """Add state constraints to block."""
        b = self.b
        stn = self.stn
        totslack = 0
        for s in stn.states:
            rhs = b.Stransfer[s]
            for t in self.TIME:
                # state capacity constraint
                b.cons.add(b.S[s, t] <= stn.C[s])
                # state mass balanace
                for i in stn.T_[s]:
                    for j in stn.K[i]:
                        rhs += stn.rho_[(i, s)]*b.A[i, j, t]
                for i in stn.T[s]:
                    for j in stn.K[i]:
                        rhs -= stn.rho[(i, s)]*b.A[i, j, t]
                if ((s, t) in self.Demand):
                    rhs -= self.Demand[s, t]
                    rhs += b.Dslack[s, t]
                    totslack += b.Dslack[s, t]
                b.cons.add(b.S[s, t] == rhs)
                b.cons.add(b.TotSlack == totslack)
                rhs = b.S[s, t]

    def add_deg_constraints(self, **kwargs):
        """Add residual life constraints to block."""
        b = self.b
        stn = self.stn
        for j in stn.units:
            rhs = b.Rtransfer[j]
            for t in self.TIME:
                # residual life balance
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        rhs += stn.D[i, j, k]*b.N[i, j, k, t]
                # rhs += b.F[j, t]
                # b.cons.add(b.R[j, t] == rhs)
                # constraints on R and F
                b.cons.add(0 <= b.R[j, t] <= stn.Rmax[j])
                # b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])
                b.cons.add(b.R[j, t] >= rhs - stn.Rmax[j]*b.M[j, t])
                b.cons.add(b.R[j, t] <= rhs)
                rhs = b.R[j, t]

    def build(self, objective="terminal", **kwargs):
        """ Only used if block is solved individually."""
        # replace D by Dmax if alpha != 0.5
        stn = self.stn
        super().build(objective=objective, **kwargs)
        # set intial values
        self.set_initial_values()
        b = self.b
        for s in stn.states:
            for t in self.TIME:
                b.cons.add(b.Dslack[s, t] == 0)

    def calc_cost_storage(self):
        """Calculate cost of storage."""
        b = self.b
        stn = self.stn

        costTotal = 0
        for t in self.TIME:
            costStorage = 0
            for s in stn.states:
                costStorage += (stn.scost[s]
                                * b.S[s, t])
            b.cons.add(b.CostStorage[t] == costStorage)
            costTotal += costStorage
        return costTotal

    def calc_cost_maintenance(self):
        """Calculate cost of maintenance."""
        b = self.b
        stn = self.stn
        costTotal = 0
        for t in self.TIME:
            costMaintenance = 0
            for j in stn.units:
                costMaintenance += ((stn.a[j] - stn.b[j])
                                    * b.M[j, t])
            b.cons.add(b.CostMaintenance[t] == costMaintenance)
            costTotal += costMaintenance
        return costTotal

    def calc_cost_wear(self):
        """Calculate wear penalty (Biondi)."""
        b = self.b
        stn = self.stn

        costTotal = 0
        for t in self.TIME:
            costWear = 0
            for j in stn.units:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        costWear += stn.D[i, j, k]*b.N[i, j, k, t]
            b.cons.add(b.CostWear[t] == costWear)
            costTotal += costWear
        return costTotal

    def add_objective(self, objective=None, **kwargs):
        """Add objective function to model."""
        assert objective is not None
        b = self.b
        stn = self.stn

        cost = self.calc_cost_storage()
        cost += self.calc_cost_maintenance()
        if objective == "biondi":
            cost += self.calc_cost_wear()
        elif objective == "terminal":
            costFinal = 0
            for j in stn.units:
                costFinal += (
                              b.R[j, self.T - self.dT]
                              / stn.Rmax[j]
                              * (stn.a[j] - stn.b[j]))
            b.cons.add(b.CostMaintenanceFinal == costFinal)
            cost += costFinal
        else:
            raise KeyError("KeyError: unknown objective %s" % objective)
        b.cons.add(b.Cost == cost)

    def add_vars(self, **kwargs):
        """Add variables to model."""
        b = self.b
        stn = self.stn

        # N[i,j,k,t] number of times task i starts on unit j in operating
        # mode k in time period t
        b.N = pyomo.Var(stn.tasks, stn.units, stn.opmodes, self.TIME,
                        domain=pyomo.NonNegativeIntegers)

        # M[j,t] 1 if unit j undergoes maintenance at time t
        b.M = pyomo.Var(stn.units, self.TIME, domain=pyomo.Boolean)

        # R[j,t] residual life time of unit j at time t
        b.R = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        # F[j,t] residual life restoration during maintenance
        b.F = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        # A[i,j,t] total amount of material undergoing task i in unit j in
        # planning time interval t
        b.A = pyomo.Var(stn.tasks, stn.units, self.TIME,
                        domain=pyomo.NonNegativeReals)

        # S[s,t] inventory of state s at time t
        b.S = pyomo.Var(stn.states, self.TIME, domain=pyomo.NonNegativeReals)

        # Q[j,t] inventory of unit j at time t
        b.Q = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        # Mode[j,k,t] 1 if unit j operates in operating mode k at time t
        b.Mode = pyomo.Var(stn.units, stn.opmodes, self.TIME,
                           domain=pyomo.Binary)

        # Variables for continuity between scheduling and planning horizon
        b.Stransfer = pyomo.Var(stn.states, domain=pyomo.NonNegativeReals)
        b.Ntransfer = pyomo.Var(stn.units, domain=pyomo.NonNegativeIntegers)
        b.Rtransfer = pyomo.Var(stn.units, domain=pyomo.NonNegativeReals)

        b.CostStorage = pyomo.Var(self.TIME, domain=pyomo.NonNegativeReals)
        b.CostMaintenance = pyomo.Var(self.TIME, domain=pyomo.NonNegativeReals)
        b.CostWear = pyomo.Var(self.TIME, domain=pyomo.NonNegativeReals)
        b.CostMaintenanceFinal = pyomo.Var(domain=pyomo.NonNegativeReals)
        b.Cost = pyomo.Var(domain=pyomo.NonNegativeReals)

        b.Dslack = pyomo.Var(stn.states, self.TIME,
                             domain=pyomo.NonNegativeReals)
        b.TotSlack = pyomo.Var(domain=pyomo.NonNegativeReals)

    def set_initial_values(self):
        """Set initial values."""
        stn = self.stn
        b = self.b

        for s in stn.states:
            b.cons.add(b.Stransfer[s] == stn.init[s])

        for j in stn.units:
            b.cons.add(b.Ntransfer[j] == 0)
            b.cons.add(b.Rtransfer[j] == stn.Rinit[j])

    def get_production_targets(self):
        b = self.b
        stn = self.stn
        cols = ["time"]
        prods = stn.products
        for p in prods:
            cols.append(p)
        df = pd.DataFrame(columns=cols)
        for t in self.TIME:
            target = []
            for p in prods:
                rhs = 0
                for i in stn.T_[p]:
                    for j in stn.K[i]:
                        rhs += stn.rho_[(i, p)]*b.A[i, j, t]()
                target.append(rhs)
            line = [t] + target
            df = df.append(pd.Series(line, index=cols),
                           ignore_index=True)
        return df

    def get_cost_maint_terminal(self, periods):
        stn = self.stn
        b = self.b
        t = (periods - 1) * self.dT
        cost = 0
        for j in stn.units:
            cost += (b.R[j, t]() / stn.Rmax[j] * (stn.a[j] - stn.b[j]))
        return cost

    def calc_p_fail(self, units=None, TP=None, periods=0, save=True, **kwargs):
        assert TP is not None
        if units is None:
            units = self.stn.units
        elif type(units) == str:
            units = set([units])
        df = pd.DataFrame(columns=units)
        for j in units:
            df[j] = calc_p_fail(self, j, self.alpha, TP, pb=True,
                                periods=periods, **kwargs)
        df["id"] = self.rid
        df["alpha"] = self.alpha

        if save:
            try:
                df2 = pd.read_pickle(self.rdir+"/"+self.prefix+"pfail.pkl")
                df2 = df2.append(df)
            except IOError:
                df2 = df
            df2.to_pickle(self.rdir+"/"+self.prefix+"pfail.pkl")
            df2.to_csv(self.rdir+"/"+self.prefix+"pfail.csv")
        return df

    def gantt(self):
        """
        Generate gantt graph.
            prefix: prepended to file name
            rdir: directory to store file
        """
        stn = self.stn
        b = self.b

        gap = self.T/400
        idx = 1
        lbls = []
        ticks = []
        # TODO: This is stupid!
        task_palette = sns.color_palette("Set2", len(stn.tasks))
        task_col = {}
        tsorted = sorted(stn.tasks)
        for task in tsorted:
            task_col.update({task: task_palette.pop()})
        mode_palette = sns.color_palette("YlOrRd", len(stn.opmodes))
        mode_col = {}
        for mode in stn.opmodes:
            mode_col.update({mode: mode_palette[stn.modeorder[mode]]})
        jsorted = sorted(stn.units)

        # number of horizontal bars to draw
        nbars = -1
        for j in jsorted:
            for i in sorted(stn.I[j]):
                nbars += 1
            nbars += 0.5
        plt.figure(figsize=(12, (nbars+1)/2))

        for j in jsorted:
            idx -= 0.5
            idx -= 1
            ticks.append(idx)
            lbls.append("{0:s}".format(j, i))
            plt.plot([self.TIME[0], self.T],
                     [idx, idx], lw=24, alpha=.3, color='y')
            for t in self.TIME:
                tau = t
                plt.axvline(t, color="black")
                for i in sorted(stn.I[j]):
                    for k in stn.O[j]:
                        if b.N[i, j, k, t]() > 0.5:
                            tauNext = (tau
                                       + b.N[i, j, k, t]()
                                       * stn.p[i, j, k])
                            plt.plot([tau, tauNext],
                                     [idx, idx], color=mode_col[k],
                                     lw=24, solid_capstyle='butt')
                            plt.plot([tau+gap, tauNext-gap],
                                     [idx, idx], color=task_col[i],
                                     lw=20, solid_capstyle='butt')
                            txt = "{0:d}".format(int(b.N[i, j, k, t]()))
                            plt.text((tau+tauNext)/2,  idx,
                                     txt,  # color=col[k],
                                     weight='bold', ha='center', va='center')
                            tau = tauNext
                if b.M[j, t]() > 0.5:
                    tauNext = tau + stn.tau[j]
                    plt.plot([tau, tauNext],
                             [idx, idx], 'k',  lw=24,  solid_capstyle='butt')
                    plt.plot([tau+gap, tauNext-gap],
                             [idx, idx], 'k', lw=20, solid_capstyle='butt')

        plt.xlim(self.TIME[0], self.T)
        plt.ylim(-nbars-0.5, 0)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls)
        plt.gca().set_xticks(self.TIME)
        plt.gca().set_xticklabels(np.round(100*self.TIME/168)/100)
        plt.savefig(self.prfx+'gantt_planning.png')
        plt.close('all')

    def trace(self):
        """
        Generate trace.
            prefix: prepended to file name
            rdir: directory to store file
        """
        # abbreviations
        b = self.b
        stn = self.stn

        oldstdout = sys.stdout
        sys.stdout = open(self.prfx + 'trace_planning.txt',
                          'w')
        print("\nStarting Conditions")
        print("\n    Initial State Inventories are:")
        for s in stn.states:
            print("        {0:10s}  {1:6.1f} kg".format(s, b.Stransfer[s]()))

        # for tracking unit assignments
        # t2go[j]['assignment'] contains the task to which unit j is currently
        # assigned
        # t2go[j]['t'] is the time to go on equipment j
        time2go = {j: {'assignment': 'None', 't': 0} for j in stn.units}

        for t in b.TIME:
            print("\nTime =", t, "hr")

            # create list of instructions
            strList = []

            for j in stn.units:
                # assign units to tasks
                fmt = ('Assign {0:s} to {1:s} for {2:.0f} batches'
                       '(Amount: {3:.1f}'
                       'kg, Mode: {4:s})')
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        amt = b.A[i, j, t]()
                        if b.N[i, j, k, t]() > 0.5:
                            strList.append(fmt.format(j, i,
                                                      b.N[i, j, k, t](),
                                                      amt, k))
                            time2go[j]['assignment'] = i
                            time2go[j]['t'] = stn.p[i, j, k]

            if len(strList) > 0:
                print()
                idx = 0
                for str in strList:
                    idx += 1
                    print('   {0:2d}. {1:s}'.format(idx, str))

            print("\n    State Inventories are now:")
            for s in stn.states:
                print("        {0:10s}  {1:6.1f} kg".format(s,
                                                            b.S[s, t]()))
            print("\n    Maintenance:")
            for j in stn.units:
                if b.M[j, t]() > 0.5:
                    print("        {0:10s}".format(j))
            # print('\n    Unit Assignments are now:')
            fmt = '        {0:s}: {1:s}, {2:.2f} kg, {3:.1f} hours to go.'
        sys.stdout = oldstdout

    def eval(self, save=True, **kwargs):
        cols = ["id", "alpha", "CostStorage", "CostMaintenance",
                "CostMainenanceFinal", "Cost", "obj", "inf", "ttot", "gap"]
        cols += self.stn.products
        units = [j for j in self.stn.units]
        dem = [0 for p in self.stn.products]
        cols += units
        if not self.inf:
            b = self.b
            gap = self.solver._gap
            if gap is None:
                gap = 0
            else:
                gap = self.solver._gap/b.Obj()*100
            cost_storage = 0
            cost_maint = 0
            cost = 0
            for t in self.TIME:
                cost_storage += b.CostStorage[t]()
                cost_maint += b.CostMaintenance[t]()
            cost_maint_final = b.CostMaintenanceFinal()
            cost += (cost_storage + cost_maint
                     + cost_maint_final)
            obj = self.b.Obj()
            pf = self.calc_p_fail(save=save, **kwargs)
            pl = []
            for j in units:
                pl.append(max(pf[j]))
            for t in self.TIME:
                for n, p in enumerate(self.stn.products):
                    if (p, t) in self.Demand:
                        dem[n] += self.Demand[(p, t)]
            df = pd.DataFrame([[self.rid, self.alpha, cost_storage,
                                cost_maint, cost_maint_final, cost, obj,
                                self.inf, self.ttot, gap]
                               + dem + pl],
                              columns=cols)
        else:
            df = pd.DataFrame([[self.rid, self.alpha, 0, 0, 0, 0, "Inf",
                                self.inf, self.ttot, "NA"] + dem
                               + [0 for j in units]],
                              columns=cols)
        if save:
            try:
                df2 = pd.read_pickle(self.rdir+"/"+self.prefix+"results.pkl")
                df2 = df2.append(df, ignore_index=True)
            except IOError:
                df2 = df
            df2.to_pickle(self.rdir+"/"+self.prefix+"results.pkl")
            df2.to_csv(self.rdir+"/"+self.prefix+"results.csv")

        return df


class blockPlanningRobust(blockPlanning):
    """Implements robust constraints for the planning horizon"""

    def __init__(self, stn, TIME, Demand, **kwargs):
        super().__init__(stn, TIME, Demand, **kwargs)

    def build(self, decisionrule="continuous", tindexed=False,
              **kwargs):
        super().build(decisionrule=decisionrule, tindexed=tindexed, **kwargs)

    def calc_nominal_R(self, tindexed=None, **kwargs):
        """Calculate R based on nominal values."""
        assert tindexed is not None
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                rhs = b.R0[j, t]
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if tindexed:
                            for tprime in self.TIME[self.TIME <= t]:
                                rhs += stn.D[i, j, k]*b.Rc[j, t, i, k, tprime]
                        else:
                            rhs += stn.D[i, j, k]*b.Rc[j, t, i, k]
                b.cons.add(b.R[j, t] == rhs)

    def calc_max_R(self, tindexed=None, **kwargs):
        """Calculate R based on nominal values."""
        assert tindexed is not None
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                rhs = b.R0[j, t]
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if tindexed:
                            for tprime in self.TIME[self.TIME <= t]:
                                rhs += (stn.D[i, j, k]
                                        * (1+stn.eps[i, j, k])
                                        * b.Rc[j, t, i, k, tprime])
                        else:
                            rhs += (stn.D[i, j, k]
                                    * (1 + stn.eps[i, j, k])
                                    * b.Rc[j, t, i, k])
                b.cons.add(b.Rmax[j, t] == rhs)

    def add_objective(self, objective=None, **kwargs):
        """Add objective function to model."""
        assert objective is not None
        b = self.b
        stn = self.stn

        cost = self.calc_cost_storage()
        cost += self.calc_cost_maintenance()
        if objective == "biondi":
            cost += self.calc_cost_wear()
        elif objective == "terminal":
            costFinal = 0
            for j in stn.units:
                costFinal += (b.Rmax[j, self.T - self.dT]
                              / stn.Rmax[j]
                              * (stn.a[j] - stn.b[j]))
            b.cons.add(b.CostMaintenanceFinal == costFinal)
        else:
            raise KeyError("KeyError: unknown objective %s" % objective)
        b.cons.add(b.Cost == cost + costFinal)

    def add_vars(self, decisionrule=None, tindexed=None, **kwargs):
        """Add variables to model."""
        assert decisionrule is not None
        assert tindexed is not None
        if decisionrule == "continuous":
            domain = pyomo.NonNegativeReals
        elif decisionrule == "integer":
            domain = pyomo.NonNegativeIntegers

        if tindexed:
            self.add_vars_tindexed(domain)
        else:
            self.add_vars_not_tindexed(domain)
        b = self.b
        stn = self.stn
        b.Rmax = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)
        super().add_vars(**kwargs)

    def add_vars_tindexed(self, domain):
        """Define affine decision rule for residual lifetime."""
        b = self.b
        stn = self.stn

        # pyomo.Variables for residual life affine decision rule
        b.R0 = pyomo.Var(stn.units, self.TIME,
                         domain=pyomo.NonNegativeReals)
        b.Rc = pyomo.Var(stn.units, self.TIME, stn.tasks, stn.opmodes,
                         self.TIME, domain=domain)

        b.R0transfer = pyomo.Var(stn.units, domain=pyomo.NonNegativeReals)
        b.Rctransfer = pyomo.Var(stn.units, stn.tasks, stn.opmodes,
                                 domain=domain)

        # Dual variables for residual life constraints
        b.ld = pyomo.Var([1, 2, 3], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)
        b.ud = pyomo.Var([1, 2, 3], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)

    def add_vars_not_tindexed(self, domain):
        """Define affine decision rule for residual lifetime."""
        b = self.b
        stn = self.stn

        # pyomo.Variables for residual life affine decision rule
        b.R0 = pyomo.Var(stn.units, self.TIME,
                         domain=pyomo.NonNegativeReals)
        b.Rc = pyomo.Var(stn.units, self.TIME, stn.tasks, stn.opmodes,
                         domain=domain)

        b.R0transfer = pyomo.Var(stn.units, domain=pyomo.NonNegativeReals)
        b.Rctransfer = pyomo.Var(stn.units, stn.tasks, stn.opmodes,
                                 domain=domain)

        # Dual variables for residual life constraints
        b.ld = pyomo.Var([1, 2, 3], stn.units, self.TIME, stn.tasks,
                         stn.opmodes,
                         domain=pyomo.NonNegativeReals)
        b.ud = pyomo.Var([1, 2, 3], stn.units, self.TIME, stn.tasks,
                         stn.opmodes,
                         domain=pyomo.NonNegativeReals)

    def define_block(self, b, decisionrule=None, **kwargs):
        # Define affine decision rule
        super().define_block(b, decisionrule=decisionrule, **kwargs)
        self.calc_nominal_R(**kwargs)
        self.calc_max_R(**kwargs)
        assert decisionrule is not None
        if decisionrule == "integer":
            self.add_int_decision_rule_cons(**kwargs)

    def add_deg_constraints(self, tindexed=None, **kwargs):
        assert tindexed is not None

        if tindexed:
            self.add_deg_constraints_tindexed()
        else:
            self.add_deg_constraints_not_tindexed()

    def add_deg_constraints_tindexed(self, **kwargs):
        """Add robust degredation constraints (tindexed: D[i,j,k,t]).

        Note:
            The residual lifetime R has a different interpretation in the
            robust implementation. It increases from 0 to Rmax instead of
            decreasing from Rmax to 0

        """
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                # constraints on R and F
                b.cons.add(0 <= b.R[j, t] <= stn.Rmax[j])
                b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])

                # inequality 1
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[1, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[1, j, t, i, k, tprime]))
                            b.cons.add(b.ud[1, j, t, i, k, tprime] -
                                       b.ld[1, j, t, i, k, tprime] >=
                                       b.Rc[j, t, i, k, tprime])
                rhs = -b.R0[j, t] + stn.Rmax[j]
                b.cons.add(lhs <= rhs)

                # inequality 2
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[2, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[2, j, t, i, k, tprime]))

                            # in the first time period R = Rtransfer
                            if (t == self.TIME[0]):
                                R0Last = b.R0transfer[j]
                                RcLast = b.Rctransfer[j, i, k]
                            else:
                                R0Last = b.R0[j, t-self.dT]
                                RcLast = b.Rc[j, t-self.dT, i, k, tprime]

                            if (tprime == t):
                                b.cons.add(b.ud[2, j, t, i, k, tprime]
                                           - b.ld[2, j, t, i, k, tprime]
                                           >= RcLast
                                           - b.Rc[j, t, i, k, tprime]
                                           + b.N[i, j, k, t])
                            else:
                                b.cons.add(b.ud[2, j, t, i, k, tprime]
                                           - b.ld[2, j, t, i, k, tprime]
                                           >= RcLast
                                           - b.Rc[j, t, i, k, tprime])
                rhs = (b.R0[j, t] - R0Last
                       + b.M[j, t]*stn.Rmax[j])
                b.cons.add(lhs <= rhs)

                # inequality 3
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps[i, j, k])
                                       * b.ud[3, j, t, i, k, tprime]
                                       - (1 - stn.eps[i, j, k])
                                       * b.ld[3, j, t, i, k, tprime]))

                            # in the first time period R = Rtransfer
                            if (t == self.TIME[0]):
                                R0Last = b.R0transfer[j]
                                RcLast = b.Rctransfer[j, i, k]
                            else:
                                R0Last = b.R0[j, t-self.dT]
                                RcLast = b.Rc[j, t-self.dT, i, k, tprime]

                            if (tprime == t):
                                b.cons.add(b.ud[3, j, t, i, k, tprime]
                                           - b.ld[3, j, t, i, k, tprime]
                                           >= -RcLast
                                           + b.Rc[j, t, i, k, tprime]
                                           - b.N[i, j, k, t])
                            else:
                                b.cons.add(b.ud[3, j, t, i, k, tprime]
                                           - b.ld[3, j, t, i, k, tprime]
                                           >= -RcLast
                                           + b.Rc[j, t, i, k, tprime])
                rhs = -b.R0[j, t] + R0Last
                b.cons.add(lhs <= rhs)

    def add_deg_constraints_not_tindexed(self, **kwargs):
        """Add robust degredation constraints (not tindexed: D[i,j,k]).

        Note:
            The residual lifetime R has a different interpretation in the
            robust implementation. It increases from 0 to Rmax instead of
            decreasing from Rmax to 0

        """
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                # constraints on R and F
                b.cons.add(0 <= b.R[j, t] <= stn.Rmax[j])
                # b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])

                # inequality 1
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[1, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[1, j, t, i, k]))
                        b.cons.add(b.ud[1, j, t, i, k] -
                                   b.ld[1, j, t, i, k] >=
                                   b.Rc[j, t, i, k])
                rhs = -b.R0[j, t] + stn.Rmax[j]
                b.cons.add(lhs <= rhs)

                # inequality 2
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[2, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[2, j, t, i, k]))

                        # in the first time period R = Rtransfer
                        if (t == self.TIME[0]):
                            R0Last = b.R0transfer[j]
                            RcLast = b.Rctransfer[j, i, k]
                        else:
                            R0Last = b.R0[j, t-self.dT]
                            RcLast = b.Rc[j, t-self.dT, i, k]

                        b.cons.add(b.ud[2, j, t, i, k]
                                   - b.ld[2, j, t, i, k]
                                   >= RcLast
                                   - b.Rc[j, t, i, k]
                                   + b.N[i, j, k, t])
                rhs = (b.R0[j, t] - R0Last
                       + b.M[j, t]*stn.Rmax[j])
                b.cons.add(lhs <= rhs)

                # inequality 3
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps[i, j, k])
                                   * b.ud[3, j, t, i, k]
                                   - (1 - stn.eps[i, j, k])
                                   * b.ld[3, j, t, i, k]))

                        # in the first time period R = Rtransfer
                        if (t == self.TIME[0]):
                            R0Last = b.R0transfer[j]
                            RcLast = b.Rctransfer[j, i, k]
                        else:
                            R0Last = b.R0[j, t-self.dT]
                            RcLast = b.Rc[j, t-self.dT, i, k]

                        b.cons.add(b.ud[3, j, t, i, k]
                                   - b.ld[3, j, t, i, k]
                                   >= -RcLast
                                   + b.Rc[j, t, i, k]
                                   - b.N[i, j, k, t])
                rhs = -b.R0[j, t] + R0Last
                b.cons.add(lhs <= rhs)

    def add_int_decision_rule_cons(self, tindexed=None, **kwargs):
        """Define affine integer decision rule for residual lifetime."""
        assert tindexed is not None
        b = self.b
        stn = self.stn
        U = 56  # TODO: make this dependent on len(sb.TIME)
        if tindexed:
            for t in self.TIME:
                for j in stn.units:
                    for i in stn.I[j]:
                        for k in stn.O[j]:
                            if (t == self.TIME[0]):
                                b.cons.add(b.Rc[j, t, i, k, t]
                                           >= b.N[i, j, k, t]
                                           + b.Rctransfer[j, i, k]
                                           - U*b.M[j, t])
                                b.cons.add(b.Rc[j, t, i, k, t]
                                           >= b.N[i, j, k, t])
                                b.cons.add(b.Rc[j, t, i, k, t]
                                           <= b.N[i, j, k, t]
                                           + b.Rctransfer[j, i, k])
                                b.cons.add(b.Rc[j, t, i, k, t]
                                           <= b.N[i, j, k, t]
                                           + U*(1 - b.M[j, t]))
                            else:
                                b.cons.add(b.Rc[j, t, i, k, t]
                                           <= b.N[i, j, k, t])
                                b.cons.add(b.Rc[j, t, i, k, t]
                                           >= b.N[i, j, k, t]
                                           - U*b.M[j, t])
                            for tprime in self.TIME[self.TIME < t]:
                                b.cons.add(b.Rc[j, t, i, k, tprime]
                                           >= b.Rc[j, t - self.dT,
                                                   i, k, tprime]
                                           - b.M[j, t]*U)
                                b.cons.add(b.Rc[j, t, i, k, tprime]
                                           <= (1 - b.M[j, t])*U)
                                b.cons.add(b.Rc[j, t, i, k, tprime]
                                           <= b.Rc[j, t - self.dT,
                                                   i, k, tprime])
        else:
            raise NotImplementedError
