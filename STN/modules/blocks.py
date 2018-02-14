#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheduling horizon of STN model with degradation.

This module implements the scheduling horizon of the STN model with degradation
by Biondi et al 2017.

The implementation of the STN is based on the implementation by Jeffrey Kantor
2017 (https://github.com/jckantor/STN-Scheduler)

"""

import pyomo.environ as pyomo
import numpy as np


class stnBlock(object):

    def __init__(self, b, stn, T_list, Demand, **kwargs):
        self.b = b
        self.stn = stn
        self.T = T_list[1]
        self.dT = T_list[2]
        TIME = range(T_list[0], self.T, self.dT)
        self.TIME = np.array(TIME)
        # self.dT = TIME[1] - TIME[0]
        # self.T = max(TIME) + self.dT
        self.Demand = Demand

        self.define_block(**kwargs)

    def define_block(self, **kwargs):
        b = self.b
        b.TIME = self.TIME
        b.T = self.T
        b.dT = self.dT
        b.cons = pyomo.ConstraintList()

        self.add_vars(**kwargs)
        self.add_unit_constraints()
        self.add_state_constraints()
        self.add_deg_constraints(**kwargs)

    def add_vars(self, **kwargs):
        raise NotImplementedError

    def add_unit_constraints(self):
        raise NotImplementedError

    def add_state_constraints(self):
        raise NotImplementedError

    def add_deg_constraints(self, **kwargs):
        raise NotImplementedError


class blockScheduling(stnBlock):

    def __init__(self, b, stn, TIME, Demand, **kwargs):
        super().__init__(b, stn, TIME, Demand, **kwargs)

    def add_unit_constraints(self):
        """Add unit constraints to block"""
        b = self.b
        stn = self.stn
        # import ipdb; ipdb.set_trace()  # noqa
        for j in stn.units:
            for t in self.TIME:
                lhs = 0
                # check if task is still running on unit j
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        if t - self.TIME[0] < stn.pinit[i, j, k]:
                            lhs += 1
                        for tprime in self.TIME[(self.TIME <= t)
                                                & (self.TIME
                                                   >= t
                                                   - stn.p[i, j, k]
                                                   + self.dT)]:
                            lhs += b.W[i, j, k, tprime]
                # check if maintenance is going on on unit j
                if t - self.TIME[0] < stn.tauinit[j]:
                    lhs += 1
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

    def add_vars(self, **kwargs):
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


class blockSchedulingRobust(blockScheduling):
    """Implements robust constraints for the scheduling horizon"""

    def __init__(self, b, stn, TIME, Demand, **kwargs):

        super().__init__(b, stn, TIME, Demand, **kwargs)

    def calc_nominal_R(self, tindexed=None):
        assert tindexed is not None
        b = self.b
        stn = self.stn

        for j in stn.units:
            for t in self.TIME:
                rhs = b.R0[j, t]
                for i in stn.I[j]:
                    for k in stn.O[j]:  # TODO: Fix this for not time indexed
                        if tindexed:
                            for tprime in self.TIME[self.TIME <= t]:
                                rhs += stn.D[i, j, k]*b.Rc[j, t, i, k, tprime]
                        else:
                            rhs += stn.D[i, j, k]*b.Rc[j, t, i, k]
                b.cons.add(b.R[j, t] == rhs)

    def calc_max_R(self, tindexed=None):
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
                                        * (1 + stn.eps)
                                        * b.Rc[j, t, i, k, tprime])
                        else:
                            rhs += (stn.D[i, j, k]
                                    * (1 + stn.eps)
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

    def define_block(self, decisionrule=None, **kwargs):
        # Define affine decision rule
        super().define_block(decisionrule=decisionrule, **kwargs)
        self.calc_nominal_R(**kwargs)
        self.calc_max_R(**kwargs)
        assert decisionrule is not None
        if decisionrule == "integer":
            self.add_int_decision_rule_cons()

    def add_deg_constraints(self, tindexed=None, **kwargs):
        assert tindexed is not None
        if tindexed:
            self.add_deg_constraints_tindexed()
        else:
            self.add_deg_constraints_not_tindexed()

    def add_deg_constraints_tindexed(self):
        """Add robust degredation constraints.

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
                # in the first time period R = Rinit
                if (t == self.TIME[0]):
                    R0Last = stn.Rinit[j]
                    RcLast = 0
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[3, j, t, i, k, tprime]
                                       - (1 - stn.eps)
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
                                    * ((1 + stn.eps)
                                       * b.ud[4, j, t, i, k, tprime]
                                       - (1 - stn.eps)
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
        """Add robust degredation constraints.

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
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps)
                                   * b.ud[1, j, t, i, k]
                                   - (1 - stn.eps)
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
                                * ((1 + stn.eps)
                                   * b.ud[2, j, t, i, k]
                                   - (1 - stn.eps)
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
                                * ((1 + stn.eps)
                                   * b.ud[3, j, t, i, k]
                                   - (1 - stn.eps)
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
                                * ((1 + stn.eps)
                                   * b.ud[4, j, t, i, k]
                                   - (1 - stn.eps)
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

    def add_int_decision_rule_cons(self):
        """Define affine decision rule for residual lifetime."""
        b = self.b
        stn = self.stn

        # add constraints for decision rule
        for t in self.TIME:
            for j in stn.units:
                for i in stn.I[j]:
                    for k in stn.O[j]:
                        b.cons.add(b.Rc[j, t, i, k, t] == b.W[i, j, k, t])
                        b.cons.add(b.Rc[j, t, i, k, t] <= 1 - b.M[j, t])
                        for tprime in self.TIME[self.TIME < t]:
                            b.cons.add(b.Rc[j, t, i, k, tprime]
                                       >= b.Rc[j, t - self.dT, i, k, tprime]
                                       - b.M[j, t])
                            b.cons.add(b.Rc[j, t, i, k, tprime]
                                       <= 1 - b.M[j, t])
                            b.cons.add(b.Rc[j, t, i, k, tprime]
                                       <= b.Rc[j, t - self.dT, i, k, tprime])


class blockPlanning(stnBlock):

    def __init__(self, b, stn, TIME, Demand, **kwargs):
        super().__init__(b, stn, TIME, Demand, **kwargs)

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
                        rhs -= stn.D[i, j, k]*b.N[i, j, k, t]
                rhs += b.F[j, t]
                b.cons.add(b.R[j, t] == rhs)
                # constraints on R and F
                b.cons.add(0 <= b.R[j, t] <= stn.Rmax[j])
                b.cons.add(b.F[j, t] <= stn.Rmax[j]*b.M[j, t])

    def add_vars(self, **kwargs):
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


class blockPlanningRobust(blockPlanning):
    """Implements robust constraints for the scheduling horizon"""

    def __init__(self, b, stn, TIME, Demand, **kwargs):
        super().__init__(b, stn, TIME, Demand, **kwargs)

    def calc_nominal_R(self, tindexed=None):
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

    def add_vars(self, decisionrule=None, tindexed=None, **kwargs):
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

    def define_block(self, decisionrule=None, **kwargs):
        # Define affine decision rule
        super().define_block(decisionrule=decisionrule, **kwargs)
        self.calc_nominal_R(**kwargs)
        assert decisionrule is not None
        # if decisionrule == "integer":
        #     self.add_int_decision_rule_cons()

    def add_deg_constraints(self, tindexed=None, **kwargs):
        assert tindexed is not None

        if tindexed:
            self.add_deg_constraints_tindexed()
        else:
            self.add_deg_constraints_not_tindexed()

    def add_deg_constraints_tindexed(self, **kwargs):
        """Add robust degredation constraints.

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
                        for tprime in self.TIME[self.TIME <= t]:
                            lhs += (stn.D[i, j, k]
                                    * ((1 + stn.eps)
                                       * b.ud[2, j, t, i, k, tprime]
                                       - (1 - stn.eps)
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
                                    * ((1 + stn.eps)
                                       * b.ud[3, j, t, i, k, tprime]
                                       - (1 - stn.eps)
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
        """Add robust degredation constraints.

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
                        lhs += (stn.D[i, j, k]
                                * ((1 + stn.eps)
                                   * b.ud[1, j, t, i, k]
                                   - (1 - stn.eps)
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
                                * ((1 + stn.eps)
                                   * b.ud[2, j, t, i, k]
                                   - (1 - stn.eps)
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
                                * ((1 + stn.eps)
                                   * b.ud[3, j, t, i, k]
                                   - (1 - stn.eps)
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

    def add_int_decision_rule_cons(self):
        """Define affine decision rule for residual lifetime."""
        b = self.b
        stn = self.stn
        U = 56  # TODO: make this dependent on len(sb.TIME)
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
                                       >= b.Rc[j, t - self.dT, i, k, tprime]
                                       - b.M[j, t]*U)
                            b.cons.add(b.Rc[j, t, i, k, tprime]
                                       <= (1 - b.M[j, t])*U)
                            b.cons.add(b.Rc[j, t, i, k, tprime]
                                       <= b.Rc[j, t - self.dT, i, k, tprime])
