#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Deterministic model of STN with degradation. Based on Biondi et al 2017.
'''
import pyomo.environ as pyomo
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import numpy as np
import dill
import sys
import csv
from blocks import (blockScheduling, blockSchedulingRobust,
                    blockPlanning, blockPlanningRobust)


class stnModel(object):
    def __init__(self):
        self.Demand = {}            # demand for products
        self.stn = StnStruct()
        self.m_list = []

    def demand(self, state, time, Demand):
        self.Demand[state, time] = Demand

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

    def calc_cost_maintenance_terminal(self):
        stn = self.stn
        m = self.model
        costMaintenance = 0
        for j in stn.units:
            for t in self.sb.TIME:
                costMaintenance += ((stn.a[j] - stn.b[j])
                                    * m.sb.M[j, t])
            for t in self.pb.TIME:
                costMaintenance += ((stn.a[j] - stn.b[j])
                                    * m.pb.M[j, t])
            costMaintenance += ((stn.Rmax[j]
                                 - m.pb.R[j, self.pb.T - self.pb.dT])
                                / stn.Rmax[j]
                                * (stn.a[j] - stn.b[j]))
        return costMaintenance

    def calc_cost_maintenance_biondi(self):
        stn = self.stn
        m = self.model
        costMaintenance = 0
        for j in stn.units:
            for t in self.sb.TIME:
                costMaintenance += (stn.a[j]*m.sb.M[j, t] -
                                    stn.b[j]*m.sb.F[j, t]/stn.Rmax[j])
            for t in self.pb.TIME:
                costMaintenance += (stn.a[j]*m.pb.M[j, t]
                                    - stn.b[j]*m.pb.F[j, t]/stn.Rmax[j])
        return costMaintenance

    def add_deg_constraints(self):
        """Add residual life continuity constraints to model."""
        stn = self.stn
        m = self.model
        for j in stn.units:
            m.cons.add(m.pb.Rtransfer[j] == m.sb.R[j, self.sb.T - self.sb.dT])

    def add_objective_terminal(self):
        """Add objective function to model."""
        m = self.model
        stn = self.stn
        m.CostStorage = pyomo.Var(domain=pyomo.NonNegativeReals)
        m.CostMaintenance = pyomo.Var(domain=pyomo.NonNegativeReals)
        m.CostWear = pyomo.Var(domain=pyomo.NonNegativeReals)

        costStorage = 0
        for s in stn.states:
            costStorage += stn.scost[s]*(m.sb.Sfin[s] +
                                         sum([m.pb.S[s, t] for t in
                                              self.pb.TIME]))
        m.cons.add(m.CostStorage == costStorage)

        m.cons.add(m.CostMaintenance == self.calc_cost_maintenance_terminal())

        m.Obj = pyomo.Objective(expr=m.CostStorage
                                + m.CostMaintenance, sense=pyomo.minimize)

    def add_objective_biondi(self):
        """Add objective function to model."""
        m = self.model
        stn = self.stn
        m.CostStorage = pyomo.Var(domain=pyomo.NonNegativeReals)
        m.CostMaintenance = pyomo.Var(domain=pyomo.NonNegativeReals)
        m.CostWear = pyomo.Var(domain=pyomo.NonNegativeReals)

        costStorage = 0
        for s in stn.states:
            costStorage += stn.scost[s]*(m.sb.Sfin[s] +
                                         sum([m.pb.S[s, t] for t in
                                              self.pb.TIME]))
        m.cons.add(m.CostStorage == costStorage)

        costWear = 0
        for j in stn.units:
            for t in self.sb.TIME:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        costWear += stn.D[i, j, k]*m.sb.W[i, j, k, t]
            for t in self.pb.TIME:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        costWear += stn.D[i, j, k]*m.pb.N[i, j, k, t]
        m.cons.add(m.CostMaintenance == self.calc_cost_maintenance_biondi())
        m.cons.add(m.CostWear == costWear)

        m.Obj = pyomo.Objective(expr=m.CostStorage
                                + m.CostMaintenance
                                + m.CostWear, sense=pyomo.minimize)
        # m.Obj = Objective(expr = m.CostStorage + m.CostMaintenance, sense =
        #                   minimize)

    def add_blocks(self, TIMEs, TIMEp, **kwargs):
        stn = self.stn
        m = self.model
        m.sb = pyomo.Block()
        self.sb = blockScheduling(m.sb, stn, TIMEs,
                                  self.Demand)
        m.pb = pyomo.Block()
        self.pb = blockPlanning(m.pb, stn, TIMEp,
                                self.Demand)

    def transfer_next_period(self):
        m = self.model
        stn = self.stn
        # import ipdb; ipdb.set_trace()  # noqa

        for s in stn.states:
            stn.init[s] = m.sb.Sfin[s]()
        for j in stn.units:
            for i in stn.I[j]:
                for k in stn.O[j]:
                    stn.pinit[i, j, k] = m.ptransfer[i, j, k]()
                    stn.Binit[i, j, k] = m.Btransfer[i, j, k]()
                    stn.tauinit[j] = m.tautransfer[j]()
            # stn.Rinit[j] = round(m.sb.R[j, self.sb.T - self.sb.dT]()*100)/100
            # stn.Rinit[j] = round(m.sb.R[j, self.sb.T - self.sb.dT]())
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

        # scheduling and planning block
        Ts = T_list[0]
        dTs = T_list[1]
        Tp = T_list[2]
        dTp = T_list[3]
        Ts_start = period * Ts
        Tp_start = (period + 1) * Ts
        Ts = Ts_start + Ts
        Tp = Tp_start + Tp
        # self.add_blocks(TIMEs, TIMEp, **kwargs)
        self.add_blocks([Ts_start, Ts, dTs], [Tp_start, Tp, dTp], **kwargs)

        # add continuity constraints to model
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints()

        # add objective function to model
        if objective == "biondi":
            self.add_objective_biondi()
        elif objective == "terminal":
            self.add_objective_terminal()
        else:
            raise KeyError("KeyError: unknown objective %s" % objective)

    def solve(self, T_list, solver='cplex', prefix='', periods=1,
              rdir='results', **kwargs):
        self.solver = pyomo.SolverFactory(solver)
        # self.solver.options['timelimit'] = 600
        self.solver.options['dettimelimit'] = 500000
        # self.solver.options['mipgap'] = 0.08
        prefix_old = prefix

        for period in range(0, periods):
            if periods > 1:
                prefix = str(period) + prefix_old
            self.build(T_list, period=period, **kwargs)
            logfile = rdir + "/" + prefix + "STN.log"
            results = self.solver.solve(self.model,
                                        tee=True,
                                        logfile=logfile)
            results.write()
            if ((results.solver.status == SolverStatus.ok) and
                (results.solver.termination_condition ==
                 TerminationCondition.optimal)):
                with open(rdir+"/"+prefix+'output.txt', 'w') as f:
                    f.write("STN Output:")
                    self.model.display(ostream=f)
                with open(rdir+"/"+prefix+'STN.pyomo', 'wb') as dill_file:
                    dill.dump(self.model, dill_file)
                self.gantt(prefix=prefix, rdir=rdir)
                self.trace(prefix=prefix, rdir=rdir)
                self.trace_planning(prefix=prefix, rdir=rdir)
                if periods > 1:
                    self.transfer_next_period()
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

    def gantt(self, prefix='', rdir=None):
        assert rdir is not None
        model = self.model
        stn = self.stn

        gap = self.sb.T/400
        idx = 1
        lbls = []
        ticks = []

        # for s in self.states:
        #     plt.plot(self.sb.TIME,
        #              [self.model.sb.S[s,t]() for t in self.sb.TIME])
        #     plt.title(s)
        #     plt.show()

        # create a list of units sorted by time of first assignment
        jstart = {j: self.sb.T+1 for j in stn.units}
        for j in stn.units:
            for i in stn.I[j]:
                for k in stn.O[j]:
                    for t in self.sb.TIME:
                        # print(self.model.W[i,j,k,t]())
                        if self.model.sb.W[i, j, k, t]() > 0:
                            jstart[j] = min(jstart[j], t)
        jsorted = [j for (j, t) in sorted(jstart.items(),  key=lambda x: x[1])]
        jsorted = sorted(stn.units)

        # number of horizontal bars to draw
        nbars = -1
        for j in jsorted:
            for i in sorted(stn.I[j]):
                nbars += 1
            nbars += 0.5
        plt.figure(figsize=(12, (nbars+1)/2))

#        print(self.model.W['Heating','Heater','Slow',0]())
#        print(O)

        for j in jsorted:
            idx -= 0.5
            idx0 = idx
            for t in self.sb.TIME:
                idx = idx0
                for i in sorted(stn.I[j]):
                    idx -= 1
                    if t == self.sb.TIME[0]:
                        ticks.append(idx)
                        plt.plot([self.sb.TIME[0], self.sb.T],
                                 [idx, idx], lw=24,
                                 alpha=.3, color='y')
                        lbls.append("{0:s} -> {1:s}".format(j, i))
                    for k in stn.O[j]:
                        if model.sb.W[i, j, k, t]() > 0.5:
                            col = {'Slow': 'green', 'Normal': 'yellow',
                                   'Fast': 'red'}  # FIX: shouldn't be explicit
                            plt.plot([t, t+stn.p[i, j, k]],
                                     [idx, idx], 'k',  lw=24,
                                     alpha=0.5, solid_capstyle='butt')
                            plt.plot([t+gap, t+stn.p[i, j, k]-gap],
                                     [idx, idx], color=col[k], lw=20,
                                     solid_capstyle='butt')
                            txt = "{0:.2f}".format(model.sb.B[i, j, k, t]())
                            plt.text(t+stn.p[i, j, k]/2, idx,  txt,
                                     weight='bold', ha='center', va='center')
                    if model.sb.M[j, t]() > 0.5:
                        plt.plot([t, t+stn.tau[j]],
                                 [idx, idx], 'k',  lw=24,
                                 alpha=0.5, solid_capstyle='butt')
                        plt.plot([t+gap, t+stn.tau[j]-gap],
                                 [idx, idx], color="grey", lw=20,
                                 solid_capstyle='butt')
                        plt.text(t+stn.tau[j]/2, idx, "Maintenance",
                                 weight='bold', ha='center', va='center')

        plt.xlim(self.sb.TIME[0], self.sb.T)
        plt.ylim(-nbars-0.5, 0)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls)
        # plt.show();
        plt.savefig(rdir+"/"+prefix+'gantt_scheduling.png')

        idx = 1
        lbls = []
        ticks = []
        # TODO: This is stupid!
        col = {'Heating': 'green', 'Reaction_1': 'yellow',
               'Reaction_3': 'red', 'Reaction_2': 'orange',
               'Separation': 'blue'}
        pat = {'Slow': 'yellow', 'Normal': 'orange', 'Fast': 'red'}
        plt.figure(figsize=(12, (nbars+1)/2))

        for j in jsorted:
            idx -= 0.5
            idx -= 1
            ticks.append(idx)
            lbls.append("{0:s}".format(j, i))
            plt.plot([self.pb.TIME[0], self.pb.T],
                     [idx, idx], lw=24, alpha=.3, color='y')
            for t in self.pb.TIME:
                tau = t
                plt.axvline(t, color="black")
                for i in sorted(stn.I[j]):
                    for k in stn.O[j]:
                        if model.pb.N[i, j, k, t]() > 0.5:
                            tauNext = (tau
                                       + model.pb.N[i, j, k, t]()
                                       * stn.p[i, j, k])
                            plt.plot([tau, tauNext],
                                     [idx, idx], color=pat[k],
                                     lw=24, solid_capstyle='butt')
                            plt.plot([tau+gap, tauNext-gap],
                                     [idx, idx], color=col[i],
                                     lw=20, solid_capstyle='butt')
                            txt = "{0:.2f}".format(model.pb.A[i, j, t]())
                            # plt.text(t+p[i, j, k]/2,  idx,
                            #          txt, color=col[k],
                            #          weight='bold', ha='center', va='center')
                            tau = tauNext
                if model.pb.M[j, t]() > 0.5:
                    tauNext = tau + stn.tau[j]
                    plt.plot([tau, tauNext],
                             [idx, idx], 'k',  lw=24,  solid_capstyle='butt')
                    plt.plot([tau+gap, tauNext-gap],
                             [idx, idx], 'k', lw=20, solid_capstyle='butt')

        plt.xlim(self.pb.TIME[0], self.pb.T)
        plt.ylim(-nbars-0.5, 0)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls)
        plt.gca().set_xticks(self.pb.TIME)
        plt.gca().set_xticklabels(np.round(100*self.pb.TIME/168)/100)
        # plt.show()
        plt.savefig(rdir+"/"+prefix+'gantt_planning.png')

        # for j in stn.units:
        #     plt.plot(self.sb.TIME, [model.sb.R[j,t]() for t in self.sb.TIME])
        #     plt.title(j)
        #     plt.show()

        for s in stn.states:
            plt.figure()
            plt.bar(self.pb.TIME/168+1,
                    [model.pb.S[s, t]() for t in self.pb.TIME])
            plt.title(s)
            # if (s,self.pb.T) in self.Demand:
            #     plt.bar(self.pb.TIME/168,
            #             [self.Demand[s,t] for t in self.pb.TIME])
            # plt.bar(self.pb.TIME/168+1,
            #         [20*model.pb.M['Reactor_1',t]() for t in self.pb.TIME])
            # plt.bar(self.pb.TIME/168+1,
            #         [10*model.pb.M['Reactor_2',t]() for t in self.pb.TIME])
            # plt.show()
            # plt.savefig(rdir+"/"+prefix+s+'.png')
            plt.close("all")

    def trace(self, prefix='', rdir=None):
        assert rdir is not None
        # abbreviations
        m = self.model
        stn = self.stn

        oldstdout = sys.stdout
        sys.stdout = open(rdir+"/"+prefix+'trace_scheduling.txt', 'w')
        print("\nStarting Conditions")
        print("\n    Initial State Inventories are:")
        for s in stn.states:
            print("        {0:10s}  {1:6.1f} kg".format(s, stn.init[s]))

        # for tracking unit assignments
        # t2go[j]['assignment'] contains the task to which unit j is currently
        # assigned
        # t2go[j]['t'] is the time to go on equipment j
        time2go = {j: {'assignment': 'None', 't': 0} for j in stn.units}

        for t in self.sb.TIME:
            print("\nTime =", t, "hr")

            # create list of instructions
            strList = []

            # first unload units
            for j in stn.units:
                time2go[j]['t'] -= self.sb.dT
                fmt = 'Transfer {0:.2f} kg from {1:s} to {2:s}'
                for i in stn.I[j]:
                    for s in stn.S_[i]:
                        for k in stn.O[j]:
                            ts = t-stn.p[i, j, k]
                            if ts >= self.sb.TIME[0]:
                                tend = max(self.sb.TIME[self.sb.TIME <= ts])
                                amt = (stn.rho_[(i, s)]
                                       * m.sb.B[i, j, k, tend]())
                                if amt > 0:
                                    strList.append(fmt.format(amt, j, s))

            for j in stn.units:
                # release units from tasks
                fmt = 'Release {0:s} from {1:s}'
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if t-stn.p[i, j, k] >= self.sb.TIME[0]:
                            tend = max(self.sb.TIME[self.sb.TIME
                                                    <= t
                                                    - stn.p[i, j, k]])
                            if m.sb.W[i, j, k, tend]() > 0:
                                strList.append(fmt.format(j, i))
                                time2go[j]['assignment'] = 'None'
                                time2go[j]['t'] = 0

                # assign units to tasks
                fmt = ('Assign {0:s} to {1:s} for {2:.2f} kg batch for {3:.1f}'
                       'hours (Mode: {4:s})')
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        amt = m.sb.B[i, j, k, t]()
                        if m.sb.W[i, j, k, t]() > 0.5:
                            strList.append(fmt.format(j, i, amt,
                                                      stn.p[i, j, k], k))
                            time2go[j]['assignment'] = i
                            time2go[j]['t'] = stn.p[i, j, k]

                # transfer from states to tasks/units
                fmt = 'Transfer {0:.2f} from {1:s} to {2:s}'
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for s in stn.S[i]:
                            amt = stn.rho[(i, s)] * m.sb.B[i, j, k, t]()
                            if amt > 0:
                                strList.append(fmt.format(amt, s, j))

                # Check if maintenance is done on unit
                fmt = 'Doing maintenance on {0:s}'
                if m.sb.M[j, t]() > 0.5:
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
                                                            m.sb.S[s, t]()))

            # print('\n    Unit Assignments are now:')
            fmt = '        {0:s}: {1:s}, {2:.2f} kg, {3:.1f} hours to go.'
            # for j in stn.units:
            #     if time2go[j]['assignment'] != 'None':
            #         print(fmt.format(j, time2go[j]['assignment'],
            #                          m.Q[j,t](), time2go[j]['t']))
            #     else:
            #         print('        {0:s} is unassigned'.format(j))

        sys.stdout = oldstdout

    def trace_planning(self, prefix='', rdir=None):
        assert rdir is not None
        # abbreviations
        m = self.model
        stn = self.stn

        oldstdout = sys.stdout
        sys.stdout = open(rdir+"/"+prefix+'trace_planning.txt', 'w')
        print("\nStarting Conditions")
        print("\n    Initial State Inventories are:")
        for s in stn.states:
            print("        {0:10s}  {1:6.1f} kg".format(s, m.sb.Sfin[s]()))

        # for tracking unit assignments
        # t2go[j]['assignment'] contains the task to which unit j is currently
        # assigned
        # t2go[j]['t'] is the time to go on equipment j
        time2go = {j: {'assignment': 'None', 't': 0} for j in stn.units}

        for t in self.pb.TIME:
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
                        amt = m.pb.A[i, j, t]()
                        if m.pb.N[i, j, k, t]() > 0.5:
                            strList.append(fmt.format(j, i,
                                                      m.pb.N[i, j, k, t](),
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
                                                            m.pb.S[s, t]()))
            print("\n    Maintenance:")
            for j in stn.units:
                if m.pb.M[j, t]() > 0.5:
                    print("        {0:10s}".format(j))
            # print('\n    Unit Assignments are now:')
            fmt = '        {0:s}: {1:s}, {2:.2f} kg, {3:.1f} hours to go.'
            # for j in stn.units:
            #     if time2go[j]['assignment'] != 'None':
            #         print(fmt.format(j, time2go[j]['assignment'],
            #                          m.Q[j,t](), time2go[j]['t']))
            #     else:
            #         cevag('        {0:f} vf hanffvtarq'.sbezng(w))
        sys.stdout = oldstdout

    def eval(self, f="STN-eval.csv"):
        m = self.model
        stn = self.stn
        with open("results/"+f, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [m.CostStorage(), m.CostMaintenance(), m.CostWear(), stn.D]
            writer.writerow(row)


class stnModelRobust(stnModel):
    def __init__(self):
        super().__init__()

    def calc_cost_maintenance_terminal(self):
        stn = self.stn
        m = self.model
        costMaintenance = 0
        for j in stn.units:
            for t in self.sb.TIME:
                costMaintenance += ((stn.a[j] - stn.b[j])
                                    * m.sb.M[j, t])
            for t in self.pb.TIME:
                costMaintenance += ((stn.a[j] - stn.b[j])
                                    * m.pb.M[j, t])
            costMaintenance += (m.pb.R[j, self.pb.T - self.pb.dT]
                                / stn.Rmax[j]
                                * (stn.a[j] - stn.b[j]))
        return costMaintenance

    def add_deg_constraints(self):
        stn = self.stn
        m = self.model
        for j in stn.units:
            m.cons.add(m.pb.R0transfer[j] == m.sb.R0[j, self.sb.T -
                                                     self.sb.dT])
            for i in stn.I[j]:
                for k in stn.O[j]:
                    rhs = 0
                    for t in self.sb.TIME:
                        rhs += m.sb.Rc[j, self.sb.T - self.sb.dT, i, k, t]
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
        self.sb = blockSchedulingRobust(m.sb, stn,
                                        np.array([t for t in TIMEs]),
                                        self.Demand,
                                        decisionrule=decisionrule)
        self.pb = blockPlanningRobust(m.pb, stn,
                                      np.array([t for t in TIMEp]),
                                      self.Demand,
                                      decisionrule=decisionrule)


class StnStruct(object):
    def __init__(self):
        # simulation objects
        self.states = set()         # set of state names
        self.tasks = set()          # set of task names
        self.units = set()          # set of unit names
        self.opmodes = set()        # set of operating mode names

        self.U = 100                # big U
        self.eps = 0.1              # Maximum deviation for uncertain D's FIX!

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
        self.D = {}                 # wear
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

    def ijkdata(self, task, unit, opmode, dur=1, wear=0, pinit=0, Binit=0):
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
