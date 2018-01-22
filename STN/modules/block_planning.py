#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning horizon of STN model with degradation.

This module implements the planning horizon of the STN model with degradation
by Biondi et al 2017.

The implementation of the STN is based on the implementation by Jeffrey Kantor
2017 (https://github.com/jckantor/STN-Scheduler)

"""

import pyomo.environ as pyomo


class blockPlanning(object):

    def __init__(self, b, stn, TIME, Demand):
        self.b = b
        self.stn = stn
        self.TIME = TIME
        self.dT = TIME[1] - TIME[0]
        self.T = max(TIME) + self.dT
        self.Demand = Demand
        self.define_planning_block()

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
                                   <= stn.U*b.Mode[j, k, t])
                b.cons.add(sum([b.Mode[j, k, t] for k in stn.O[j]]) == 1)
                lhs = 0  # set lhs to zero for next time period

    def add_state_constraints(self):
        """Add state constraints to block."""
        b = self.b
        stn = self.stn
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
                b.cons.add(b.S[s, t] == rhs)
                rhs = b.S[s, t]

    def add_deg_constraints(self):
        """Add residual life constraints to block."""
        b = self.b
        stn = self.stn
        for j in stn.units:
            rhs = b.Rtransfer[j]
            for t in self.TIME:
                # residual life balance
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        rhs -= stn.D[i, j, k]*b.N[i, j, k, t]
                rhs += b.F[j, t]
                b.cons.add(b.R[j, t] == rhs)
                # constraints on R and F
                b.cons.add(0 <= b.R[j, t] <= stn.Rmax[j])
                b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])

    def define_planning_block(self):
        b = self.b
        stn = self.stn
        b.cons = pyomo.ConstraintList()

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

        # Add constraints
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints()


class blockPlanningRobust(blockPlanning):
    """Implements robust constraints for the planning horizon."""

    def __init__(self, b, stn, TIME, Demand):
        blockPlanning.__init__(self, b, stn, TIME, Demand)

    def add_deg_constraints(self):
        """Adds robust degradation constraints to block."""
        b = self.b
        stn = self.stn

        # Dual variables for residual life constraints
        b.ld = pyomo.Var([1, 2, 3], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)
        b.ud = pyomo.Var([1, 2, 3], stn.units, self.TIME, stn.tasks,
                         stn.opmodes, self.TIME,
                         domain=pyomo.NonNegativeReals)

        # pyomo.Variables for residual life affine decision rule
        b.Rc = pyomo.Var(stn.units, self.TIME, stn.tasks, stn.opmodes,
                         self.TIME, domain=pyomo.NonNegativeReals)
        b.R0 = pyomo.Var(stn.units, self.TIME, domain=pyomo.NonNegativeReals)

        for j in stn.units:
            for t in self.TIME:
                # constraints on R and F
                # do these need to stay??
                # b.cons.add(0 <= b.R[j,t] <= stn.Rmax[j])
                # b.cons.add(b.F[j,t] <= stn.Rmax[j]*b.M[j,t])

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
                                       b.ld[1, j, t, i, k, tprime] >=
                                       b.Rc[j, t, i, k, tprime])
                rhs = -b.R0[j, t] + stn.Rmax[j]
                b.cons.add(lhs <= rhs)

                # inequality 2
                lhs = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[2, j, t, i, k, tprime]
                                       - (1 - stn.eps)
                                       * b.ld[2, j, t, i, k, tprime]))

                            # in the first time period R = Rtransfer
                            if (t == self.TIME[0]):
                                R0Last = b.Rtransfer[j]
                                RcLast = 0
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
                        for tprime in self.TIME:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[3, j, t, i, k, tprime]
                                       - (1 - stn.eps)
                                       * b.ld[3, j, t, i, k, tprime]))

                            # in the first time period R = Rtransfer
                            if (t == self.TIME[0]):
                                R0Last = b.Rtransfer[j]
                                RcLast = 0
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
