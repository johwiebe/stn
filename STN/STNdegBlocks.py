#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:42:12 2017

@author: jeff
"""

from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np

class STN(object):
    def __init__(self):
        # simulation objects
        self.states = set()         # set of state names
        self.tasks = set()          # set of task names
        self.units = set()          # set of unit names
        self.opmodes = set()        # set of operating mode names
        self.TIMEs = []             # time grid scheduling
        self.TIMEp = []             # time grid planning
        self.Demand = {}            # demand for products
        self.dTs = 1                # scheduling time period
        self.dTp = 1                # planning time period
        self.Ts = 10                # scheduling time period
        self.Tp = 10                # planning time period

        self.U = 10000              # big U

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
        self.I = {}                 # sets of tasks performed by each unit
        self.O = {}                 # sets of operating modes in which each unit can operate
        self.tau = {}               # time taken to do maintenance for each unit
        self.a = {}                 # fixed maintenance cost for each unit
        self.b = {}                 # maintenance discount for each unit

        # dictionaries indexed by (task,state)
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

        # dictionaries indexed by (task,task)
        self.changeoverTime = {}        # switch over times required for task1 -> task2

    # defines states as .state(name, capacity, init)
    def state(self, name, capacity = float('inf'), init = 0, price = 0,):
        self.states.add(name)       # add to the set of states
        self.C[name] = capacity     # state capacity
        self.init[name] = init      # state initial value
        self.T[name] = set()        # tasks which feed this state (inputs)
        self.T_[name] = set()       # tasks fed from this state (outputs)
        self.price[name] = price    # per unit price of each state

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
        self.rho[(task,state)] = rho
        self.T[state].add(task)
        
    def tsArc(self, task, state, rho=1, dur=1):
        if state not in self.states:
            self.state(state)
        if task not in self.tasks:
            self.task(task)
        self.S_[task].add(state)
        self.T_[state].add(task)
        self.rho_[(task,state)] = rho
        self.P[(task,state)] = dur
#        self.p[task] = max(self.p[task],dur)

    def unit(self, unit, task, Bmin = 0, Bmax = float('inf'), cost = 0, vcost = 0, tm = 0, rmax = 0, rinit = 0, a = 0, b = 0):
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
        self.Bmin[(task,unit)] = Bmin
        self.Bmax[(task,unit)] = Bmax
        self.cost[(task,unit)] = cost
        self.vcost[(task,unit)] = vcost
        self.tau[unit] = max(tm, self.tau[unit])
        self.Rmax[unit] = max(rmax, self.Rmax[unit])
        self.Rinit[unit] = max(rinit, self.Rinit[unit])
        self.a[unit] = max(a, self.a[unit])
        self.b[unit] = max(b, self.b[unit])

    def opmode(self, opmode):
        if opmode not in self.opmodes:
            self.opmodes.add(opmode)
    
    def ijkdata(self, task, unit, opmode, dur = 1, wear = 0):
        if opmode not in self.O[unit]:
            self.O[unit].add(opmode)
        self.p[task, unit, opmode] = dur
        self.D[task, unit, opmode] = wear
    
    def changeover(self, task1, task2, dur):
        self.changeoverTime[(task1,task2)] = dur

    def demand(self, state, time, Demand):
        self.Demand[state, time] = Demand

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
        for (task,state) in sorted(self.rho.keys()):
            print('    {0:s} -> {1:s}:'.format(state,task))
            print('        rho:', self.rho[(task,state)])

        print('\nTask -> State Arcs')  
        for (task,state) in sorted(self.rho_.keys()):
            print('    {0:s} -> {1:s}:'.format(task,state))
            print('        rho_:', self.rho_[(task,state)])
            print('           P:', self.P[(task,state)])

    # build model
    def build(self, TIMEs, TIMEp):
        self.TIMEs = np.array([t for t in TIMEs])
        self.TIMEp = np.array([t for t in TIMEp])
        self.dTs = self.TIMEs[2] - self.TIMEs[1]
        self.dTp = self.TIMEp[2] - self.TIMEp[1]
        self.Ts = max(self.TIMEs)
        self.Tp = max(self.TIMEp)
        self.model = ConcreteModel()
        m = self.model
        m.cons = ConstraintList()
        
        # define scheduling block
        def scheduling_block(b):
            b.cons = ConstraintList()
            b.TIME = self.TIMEs
            b.T = self.Ts
            b.dT = self.dTs

            # W[i,j,k,t] 1 if task i starts in unit j and operating mode k at time t
            b.W = Var(self.tasks, self.units, self.opmodes, b.TIME, domain=Boolean)

            # M[j,t] 1 if unit j undergoes maintenance at time t
            b.M = Var(self.units, b.TIME, domain=Boolean)

            # R[j,t] residual life time of unit j at time t
            b.R = Var(self.units, b.TIME, domain=NonNegativeReals)

            # F[j,t] residual life restoration during maintenance
            b.F = Var(self.units, b.TIME, domain=NonNegativeReals)
            
            # B[i,j,k,t] size of batch assigned to task i in unit j at time t
            b.B = Var(self.tasks, self.units, self.opmodes, b.TIME, domain=NonNegativeReals)

            # S[s,t] inventory of state s at time t
            b.S = Var(self.states, b.TIME, domain=NonNegativeReals)
            
            # Q[j,t] inventory of unit j at time t
            b.Q = Var(self.units, b.TIME, domain=NonNegativeReals)
        
            # unit constraints
            for j in self.units:
                rhs = 0
                for t in b.TIME:
                    # a unit can only be allocated to one task 
                    lhs = 0
                    for i in self.I[j]:
                        for k in self.O[j]:
                            for tprime in b.TIME[(b.TIME <= t) & (b.TIME >= t-self.p[i,j,k]+1)]:
                                lhs += b.W[i,j,k,tprime]
                            for tprime in b.TIME[(b.TIME <= t) & (b.TIME >= t-self.tau[j]+1)]:
                                lhs += b.M[j,tprime]
                    b.cons.add(lhs <= 1)
                    
                    # capacity constraints (see Konkili, Sec. 3.1.2)
                    for i in self.I[j]:
                        for k in self.O[j]:
                            b.cons.add(b.W[i,j,k,t]*self.Bmin[i,j] <= b.B[i,j,k,t])
                            b.cons.add(b.B[i,j,k,t] <= b.W[i,j,k,t]*self.Bmax[i,j])
                    
                    # unit mass balance
#                    rhs += sum([b.B[i,j,k,t] for i in self.I[j] for k in self.O[j]])
#                    for i in self.I[j]:
#                        for s in self.S_[i]:
#                            if t >= self.P[(i,s)]:
#                                rhs -= self.rho_[(i,s)]*sum([b.B[i,j,k,max(b.TIME[b.TIME <= t-self.P[(i,s)]])] for k in self.O[j]])
#                    b.cons.add(b.Q[j,t] == rhs)
#                    rhs = b.Q[j,t]
                    
                    # switchover time constraints
#                    for (i1,i2) in self.changeoverTime.keys():
#                        if (i1 in self.I[j]) and (i2 in self.I[j]):
#                            for t1 in b.TIME[b.TIME <= (self.H - self.p[i1])]:
#                                for t2 in b.TIME[(b.TIME >= t1 + self.p[i1])
#                                        & (b.TIME < t1 + self.p[i1] + self.changeoverTime[(i1,i2)])]: 
#                                    b.cons.add(sum([b.W[i1,j,k,t1] for k in self.O[j]]) + sum([b.W[i2,j,k,t2] for k in self.O[k]]) <= 1)

                    
                    # terminal condition
                    # b.cons.add(b.Q[j,b.T] == 0)

            # state constraints
            for s in self.states:
                rhs = self.init[s]
                for t in b.TIME:
                    # state capacity constraint
                    b.cons.add(b.S[s,t] <= self.C[s])
                    # state mass balanace
                    for i in self.T_[s]:
                        for j in self.K[i]:
                            for k in self.O[j]:
                                if t >= self.P[(i,s)] + self.p[i,j,k]:
                                    rhs += self.rho_[(i,s)]*b.B[i,j,k,max(b.TIME[b.TIME <= t-self.p[i,j,k]-self.P[(i,s)]])]
                    for i in self.T[s]:
                        for j in self.K[i]:
                            for k in self.O[j]:
                                rhs -= self.rho[(i,s)]*b.B[i,j,k,t]
                    if (s,t-b.dT) in self.Demand:
                        rhs -= self.Demand[s,t-b.dT]
                    b.cons.add(b.S[s,t] == rhs)
                    rhs = b.S[s,t]

            # residual life constraints
            for j in self.units:
                rhs = self.Rinit[j]
                for t in b.TIME:
                    # constraints on F[j,t] and R[j,t]
                    b.cons.add(b.F[j,t] <= self.Rmax[j]*b.M[j,t])
                    b.cons.add(b.F[j,t] <= self.Rmax[j] - rhs)
                    b.cons.add(b.F[j,t] >= self.Rmax[j]*b.M[j,t] - rhs)
                    b.cons.add(0 <= b.R[j,t] <= self.Rmax[j])
                    # residual life balance
                    for i in self.I[j]:
                        for k in self.O[j]:
                            rhs -= self.D[i,j,k]*b.W[i,j,k,t]
                    rhs += b.F[j,t]
                    b.cons.add(b.R[j,t] == rhs)

        # define planning block
        def planning_block(b):
            b.cons = ConstraintList()
            b.TIME = self.TIMEp
            b.T = self.Tp
            b.dT = self.dTp
            m = b.parent_block()
            import ipdb; ipdb.set_trace()
            
            # N[i,j,k,t] number of times task i starts on unit j in operating mode k in time period t
            b.N = Var(self.tasks, self.units, self.opmodes, b.TIME, domain = Boolean)

            # M[j,t] 1 if unit j undergoes maintenance at time t
            b.M = Var(self.units, b.TIME, domain=Boolean)

            # R[j,t] residual life time of unit j at time t
            b.R = Var(self.units, b.TIME, domain=NonNegativeReals)

            # F[j,t] residual life restoration during maintenance
            b.F = Var(self.units, b.TIME, domain=NonNegativeReals)
            
            # A[i,j,t] total amount of material undergoing task i in unit j in planning time interval t
            b.A = Var(self.tasks, self.units, b.TIME, domain=NonNegativeReals)
            
            # S[s,t] inventory of state s at time t
            b.S = Var(self.states, b.TIME, domain=NonNegativeReals)
            
            # Q[j,t] inventory of unit j at time t
            b.Q = Var(self.units, b.TIME, domain=NonNegativeReals)

            # Mode[j,k,t] 1 if unit j operates in operating mode k at time t
            b.Mode = Var(self.units, self.opmodes, b.TIME)

            # planning horizon constraints
            # unit constraints
            for j in self.units:
                rhs = m.Ntransfer[j]
                for t in b.TIME:
                    # a unit can only be allocated to one task
                    lhs = 0
                    for i in self.I[j]:
                        for k in self.O[j]:
                            lhs += b.N[i,j,k,t]*self.p[i,j,k]
                    lhs += b.M[j,t]*self.tau[j]
                    b.cons.add(lhs <= b.dT)

                    # capacity constraints (see Konkili, Sec. 3.1.2)
                    for i in self.I[j]:
                        b.cons.add(sum([b.N[i,j,k,t] for k in self.O[j]])*self.Bmin[i,j] <= b.A[i,j,t])
                        b.cons.add(b.A[i,j,t] <= sum([b.N[i,j,k,t] for k in self.O[j]])*self.Bmax[i,j])

                    # operating mode constraints
                    for i in self.I[j]:
                        for k in self.O[j]:
                            b.cons.add(b.N[i,j,k,t] <= self.U*b.Mode[j,k,t])
                    b.cons.add(sum([b.Mode[j,k,t] for k in self.O[j]]) == 1)

            # state constraints
            for s in self.states:
                rhs = m.Stransfer[s]
                for t in b.TIME:
                    # state capacity constraint
                    b.cons.add(b.S[s,t] <= self.C[s])
                    # state mass balanace
                    for i in self.T_[s]:
                        for j in self.K[i]:
                            rhs += self.rho_[(i,s)]*b.A[i,j,t]
                    for i in self.T[s]:
                        for j in self.K[i]:
                            rhs -= self.rho[(i,s)]*b.A[i,j,t]
                    if (s,t) in self.Demand:
                        rhs -= self.Demand[s,t]
                    b.cons.add(b.S[s,t] == rhs)
                    rhs = b.S[s,t]

            # residual life constraints
            for j in self.units:
                rhs = m.Rtransfer[j] 
                for t in b.TIME:
                    # residual life balance
                    for i in self.I[j]:
                        for k in self.O[j]:
                            rhs -= self.D[i,j,k]*b.N[i,j,k,t]
                    rhs += b.F[j,t]
                    b.cons.add(b.R[j,t] == rhs)
                    # constraints on R and F
                    b.cons.add(0 <= b.R[j,t] <= self.Rmax[j])
                    b.cons.add(b.F[j,t] <= self.Rmax[j]*b.M[j,t])

        m.Sfin = Var(self.states, domain=NonNegativeReals)
        m.Stransfer = Var(self.states, domain=NonNegativeReals)
        m.Ntransfer = Var(self.units, domain=NonNegativeIntegers)
        m.Rtransfer = Var(self.units, domain=NonNegativeReals)

        m.sb = Block(rule=scheduling_block)
        m.pb = Block(rule=planning_block)

        # connection between scheduling and planning
        # continuity of state storage
        for s in self.states:
            rhs = m.sb.S[s,self.Ts]
            for i in self.T_[s]:
                for j in self.K[i]:
                    for k in self.O[j]:
                        rhs += self.rho_[(i,s)]*m.sb.B[i,j,k,self.Ts + self.dTs
                                - self.p[i,j,k]]
            if (s,self.Ts) in self.Demand:
                self.Demand[s,self.Ts]
            m.cons.add(m.Sfin[s] == rhs)
            m.cons.add(0 <= m.Sfin[s] <= self.C[s])
            for i in self.T_[s]:
                for j in self.K[i]:
                    for k in self.O[j]:
                        for tprime in self.TIMEs[(self.TIMEs <= self.Ts) &
                                (self.TIMEs >= self.Ts-self.p[i,j,k]+2)]:
                            rhs += self.rho_[(i,s)]*m.sb.B[i,j,k,tprime]
            m.cons.add(m.Stransfer[s] == rhs)

        # continuity of allocation
        for j in self.units:
            rhs = 0
            for i in self.I[j]:
                for k in self.O[j]:
                    for tprime in self.TIMEs[(self.TIMEs <= self.Ts) &
                            (self.TIMEs >= self.Ts-self.p[i,j,k]+2)]:
                        rhs += m.sb.W[i,j,k,tprime]*(self.p[i,j,k] - (self.Ts - tprime
                            + 1))
            for tprime in self.TIMEs[(self.TIMEs <= self.Ts) & (self.TIMEs >=
                self.Ts-self.tau[j]+2)]:
                rhs += m.sb.M[j,tprime]*(self.tau[j] - (self.Ts - tprime + 1))
            m.cons.add(m.Ntransfer[j] == rhs)

        # continuity of residual lifetime
        for j in self.units:
            m.cons.add(m.Rtransfer[j] == m.sb.R[j,self.Ts])

        # objective function
        m.CostStorage = Var(domain=NonNegativeReals)
        m.CostMaintenance = Var(domain=NonNegativeReals)
        m.CostWear = Var(domain=NonNegativeReals)

        costStorage = 0
        for s in self.states:
            costStorage += self.scost[s]*(Sfin[s] + sum([m.pb.S[s,t] for t in
                self.TIMEp]))
        m.cons.add(m.CostStorage == costStorage)

        costMaintenance = 0
        costWear = 0
        for j in self.units:
            for t in self.TIMEs:
                costMaintenance += (self.a[j]*m.sb.M[j,t] -
                                   self.b[j]*m.sb.F[j,t]/self.Rmax[j])
                for i in self.I[j]:
                    for k in self.O[j]:
                        costWear += self.D[i,j,k]*m.sb.W[i,j,k,t]
            for t in self.TIMEp:
                costMaintenance += (self.a[j]*m.pb.M[j,t] -
                                   self.b[j]*m.pb.F[j,t]/self.Rmax[j])
                for i in self.I[j]:
                    for k in self.O[j]:
                        costWear += self.D[i,j,k]*m.pb.N[i,j,k,t]
        m.cons.add(m.CostMaintenance == costMaintenance)
        m.cons.add(m.CostWear == costWear)

        
        m.cons.add(m.Value == sum([self.price[s]*m.sb.S[s,self.Ts] for s in self.states]))
        m.cons.add(m.Cost == sum([self.cost[(i,j)] * m.sb.W[i,j,k,t] + self.vcost[(i,j)] * m.sb.B[i,j,k,t]
                                   for i in self.tasks for j in self.K[i] for k in self.O[j] for t in self.TIMEs]))
        m.Obj = Objective(expr = m.Value - m.Cost, sense = maximize)


    def solve(self, solver='cplex'):
        self.solver = SolverFactory(solver)
#        self.solver.options['tmlim'] = 600
        self.solver.solve(self.model, tee=True).write()

    def gantt(self):
        model = self.model
        C = self.C
        H = self.Ts
        I = self.I
        p = self.p
        O = self.O

        import ipdb; ipdb.set_trace()
        gap = H/400
        idx = 1
        lbls = []
        ticks = []
        
        for s in self.states:
            plt.plot(self.TIMEs, [self.model.sb.S[s,t]() for t in self.TIMEs])
            plt.show()

        #for j in self.units:
        #    plt.plot(self.TIMEs, [self.model.M[j,t]() for t in self.TIMEs])
        #    plt.show()
        
        # create a list of units sorted by time of first assignment
        jstart = {j:H+1 for j in self.units}
        for j in self.units:
            for i in I[j]:
                for k in O[j]: 
                    for t in self.TIMEs:
                        #print(self.model.W[i,j,k,t]())
                        if self.model.sb.W[i,j,k,t]() > 0:
                            jstart[j] = min(jstart[j],t)
        jsorted = [j for (j,t) in sorted(jstart.items(), key=lambda x: x[1])]

        # number of horizontal bars to draw
        nbars = -1
        for j in jsorted:
            for i in sorted(I[j]):
                nbars += 1
            nbars += 0.5
        plt.figure(figsize=(12,(nbars+1)/2))

#        print(self.model.W['Heating','Heater','Slow',0]())
#        print(O)
        
        for j in jsorted:
            idx -= 0.5
            for i in sorted(I[j]):
                idx -= 1
                ticks.append(idx)
                lbls.append("{0:s} -> {1:s}".format(j,i))
                plt.plot([0,H],[idx,idx],lw=24,alpha=.3,color='y')
                for t in self.TIMEs:
                    for k in O[j]:
                        if model.sb.W[i,j,k,t]() > 0:
                            plt.plot([t,t+p[i,j,k]], [idx,idx],'k', lw=24, alpha=0.5, solid_capstyle='butt')
                            plt.plot([t+gap,t+p[i,j,k]-gap], [idx,idx],'b', lw=20, solid_capstyle='butt')
                            txt = "{0:.2f}".format(model.sb.B[i,j,k,t]())
                            col = {'Slow': 'green', 'Normal': 'yellow', 'Fast': 'red'}
                            plt.text(t+p[i]/2, idx, txt, color=col[k], weight='bold', ha='center', va='center')
        plt.xlim(0,self.Ts)
        plt.ylim(-nbars-0.5,0)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls);
        plt.show();

    def trace(self):
        # abbreviations
        model = self.model
        TIMEs = self.TIMEs
        dT = np.mean(np.diff(TIMEs))
        
        print("\nStarting Conditions")
        print("\n    Initial State Inventories are:")
        for s in self.states:
            print("        {0:10s}  {1:6.1f} kg".format(s,self.init[s]))
        
        # for tracking unit assignments
        # t2go[j]['assignment'] contains the task to which unit j is currently assigned
        # t2go[j]['t'] is the time to go on equipment j
        time2go = {j:{'assignment':'None', 't':0} for j in self.units}
        
        for t in TIMEs:
            print("\nTime =",t,"hr")
            
            # create list of instructions
            strList = []
            
            # first unload units 
            for j in self.units:
                time2go[j]['t'] -= dT
                fmt = 'Transfer {0:.2f} kg from {1:s} to {2:s}'
                for i in self.I[j]:  
                    for s in self.S_[i]:
                        ts = t-self.P[(i,s)]
                        if ts >= 0:
                            amt = self.rho_[(i,s)] * model.B[i,j, max(TIMEs[TIMEs <= ts])]()
                            if amt > 0:
                                strList.append(fmt.format(amt,j,s))
                                
            for j in self.units:
                # release units from tasks
                fmt = 'Release {0:s} from {1:s}'
                for i in self.I[j]:
                    if t-self.p[i] >= 0:
                        if model.W[i,j,max(TIMEs[TIMEs <= t-self.p[i]])]() > 0:
                            strList.append(fmt.format(j,i))
                            time2go[j]['assignment'] = 'None'
                            time2go[j]['t'] = 0
                            
                # assign units to tasks
                fmt = 'Assign {0:s} to {1:s} for {2:.2f} kg batch for {3:.1f} hours'
                for i in self.I[j]:
                    amt = model.B[i,j,t]()
                    if amt > 0:
                        strList.append(fmt.format(j,i,amt,self.p[i]))
                        time2go[j]['assignment'] = i
                        time2go[j]['t'] = self.p[i]
                        
                # transfer from states to tasks/units
                fmt = 'Transfer {0:.2f} from {1:s} to {2:s}'
                for i in self.I[j]:
                    for s in self.S[i]:
                        amt = self.rho[(i,s)] * model.B[i,j,t]()
                        if amt > 0:
                            strList.append(fmt.format(amt, s, j))

            if len(strList) > 0:
                print()
                idx = 0
                for str in strList:
                    idx += 1
                    print('   {0:2d}. {1:s}'.format(idx,str))
                    
            print("\n    State Inventories are now:")
            for s in self.states:
                print("        {0:10s}  {1:6.1f} kg".format(s,model.S[s,t]()))
            
            print('\n    Unit Assignments are now:')
            fmt = '        {0:s}: {1:s}, {2:.2f} kg, {3:.1f} hours to go.'
            for j in self.units:
                if time2go[j]['assignment'] != 'None':
                    print(fmt.format(j, time2go[j]['assignment'], 
                                     model.Q[j,t](), time2go[j]['t']))
                else:
                    print('        {0:s} is unassigned'.format(j))
