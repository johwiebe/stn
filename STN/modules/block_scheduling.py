#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheduling horizon of STN model with degradation.

This module implements the scheduling horizon of the STN model with degradation
by Biondi et al 2017.

The implementation of the STN is based on the implementation by Jeffrey Kantor
2017 (https://github.com/jckantor/STN-Scheduler)

"""

import pyomo.environ as pyomo


class blockScheduling(object):

    def __init__(self, b, stn, TIME, Demand):
        self.b = b
        self.stn = stn
        self.TIME = TIME
        self.dT = TIME[1] - TIME[0]
        self.T = max(TIME) + self.dT
        self.Demand = Demand
        self.define_scheduling_block()

    def add_unit_constraints(self):
        """Add unit constraints to block"""
        b = self.b
        stn = self.stn
        for j in stn.units:
            for t in self.TIME:
                lhs = 0
                # check if task is still running on unit j
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[(self.TIME <= t)
                                                & (self.TIME
                                                   >= t
                                                   - stn.p[i, j, k]
                                                   + self.dT)]:
                            lhs += b.W[i, j, k, tprime]
                # check if maintenance is going on on unit j
                for tprime in self.TIME[(self.TIME <= t)
                                        & (self.TIME
                                           >= t
                                           - stn.tau[j]
                                           + self.dT)]:
                    lhs += b.M[j, tprime]
                # a unit can only be allocated to one task
                b.cons.add(lhs <= 1)

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
                b.cons.add(b.S[s, t] <= stn.C[s])
                # state mass balanace
                for i in stn.T_[s]:
                    for j in stn.K[i]:
                        for k in stn.O[j]:
                            if t >= stn.P[(i, s)] + stn.p[i, j, k]:
                                tprime = max(self.TIME[self.TIME
                                                       <= t
                                                       - stn.p[i, j, k]
                                                       - stn.P[(i, s)]])
                                rhs += stn.rho_[(i, s)]*b.B[i, j, k,
                                                            tprime]
                for i in stn.T[s]:
                    for j in stn.K[i]:
                        for k in stn.O[j]:
                            rhs -= stn.rho[(i, s)]*b.B[i, j, k, t]
                if (s, t - self.dT) in self.Demand:
                    rhs -= self.Demand[s, t - b.dT]
                b.cons.add(b.S[s, t] == rhs)
                rhs = b.S[s, t]

    def add_deg_constraints(self):
        """Add residual life constraints to block"""
        b = self.b
        stn = self.stn
        for j in stn.units:
            rhs = stn.Rinit[j]
            for t in self.TIME:
                # constraints on F[j,t] and R[j,t]
                b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])
                b.cons.add(b.F[j, t] <= stn.Rmax[j] - rhs)
                b.cons.add(b.F[j, t] >= stn.Rmax[j]*b.M[j, t] - rhs)
                b.cons.add(0 <= b.R[j, t] <= stn.Rmax[j])
                # residual life balance
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        rhs -= stn.D[i, j, k]*b.W[i, j, k, t]
                rhs += b.F[j, t]
                b.cons.add(b.R[j, t] == rhs)

    def define_scheduling_block(self):
        b = self.b
        stn = self.stn
        b.cons = pyomo.ConstraintList()

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

        # Add constraints to block
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints()


class blockSchedulingRobust(blockScheduling):
    """Implements robust constraints for the scheduling horizon"""

    def __init__(self, b, stn, TIME, Demand):
        blockScheduling.__init__(self, b, stn, TIME, Demand)

    def add_deg_constraints(self):
        """Add robust degredation constraints.

        Note:
            The residual lifetime R has a different interpretation in the
            robust implementation.

        """

        b = self.b
        stn = self.stn
        # Dual variables for residual life constraints
        b.ld = pyomo.Var([1, 2, 3, 4], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)
        b.ud = pyomo.Var([1, 2, 3, 4], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)

        # pyomo.Variables for residual life affine decision rule
        b.Rc = pyomo.Var(stn.units, self.TIME, stn.tasks, stn.opmodes,
                         self.TIME, domain=pyomo.NonNegativeReals)
        b.R0 = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        for j in stn.units:
            RcLast = stn.Rinit[j]
            for t in self.TIME:
                # constraints on F[j,t] and R[j,t]
                # b.cons.add(b.F[j,t] <= stn.Rmax[j]*b.M[j,t])
                # b.cons.add(b.F[j,t] <= stn.Rmax[j] - rhs)
                # b.cons.add(b.F[j,t] >= stn.Rmax[j]*b.M[j,t] - rhs)
                # b.cons.add(b.R[j, t] <= stn.Rmax[j])
                # b.cons.add(stn.Rmax[j]*b.M[j, t] <= b.R[j, t])
                # residual life balance
                # inequality 1
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[1, j, t, i, k, tprime]
                                       - (1 - stn.eps)
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
                                    * ((1 + stn.eps)
                                       * b.ud[2, j, t, i, k, tprime]
                                       - (1 - stn.eps)
                                       * b.ld[2, j, t, i, k, tprime]))
                            b.cons.add(b.ud[2, j, t, i, k, tprime] -
                                       b.ld[2, j, t, i, k, tprime] >=
                                       b.Rc[j, t, i, k, tprime])
                rhs = -b.R0[j, t] + stn.Rmax[j]*(1 - b.M[j, t])
                # rhs = -b.R0[j,t] + stn.Rmax[j]
                b.cons.add(lhs <= rhs)

                # inequality 3
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[3, j, t, i, k, tprime]
                                       - (1 - stn.eps)
                                       * b.ld[3, j, t, i, k, tprime]))

                            # in the first time period R = Rinit
                            if (t == self.TIME[0]):
                                R0Last = stn.Rinit[j]
                                RcLast = 0
                            else:
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
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[4, j, t, i, k, tprime]
                                       - (1 - stn.eps)
                                       * b.ld[4, j, t, i, k, tprime]))

                            # in the first time period R = Rinit
                            if (t == self.TIME[0]):
                                R0Last = stn.Rinit[j]
                                RcLast = 0
                            else:
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
