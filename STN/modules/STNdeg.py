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
        self.H = 0                  # planning time period

        # dictionaries indexed by task name
        self.S = {}                 # sets of states feeding each task (inputs)
        self.S_ = {}                # sets of states fed by each task (outputs)
        self.K = {}                 # sets of units capable of each task 
#        self.p = {}                 # task durations
        
        # dictionaries indexed by state name
        self.T = {}                 # tasks fed from each state (task output)
        self.T_ = {}                # tasks feeding each state (task inputs)
        self.C = {}                 # capacity of each task
        self.init = {}              # initial level
        self.price = {}             # prices of each state
        
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
            
    def build(self, TIMEs):
        
        self.TIMEs = np.array([t for t in TIMEs])
        self.H = max(self.TIMEs)
        self.model = ConcreteModel()
        m = self.model
        m.cons = ConstraintList()
        
        # W[i,j,k,t] 1 if task i starts in unit j and operating mode k at time t
        m.W = Var(self.tasks, self.units, self.opmodes, self.TIMEs, domain=Boolean)

        # N[i,j,k,t] number of times task i starts on unit j in operating mode k in time period t
        m.N = Var(self.task, self.units, self.opmodes, self.TIMEp, domain = Boolean)

        # M[j,t] 1 if unit j undergoes maintenance at time t
        m.M = Var(self.units, self.TIMEs, domain=Boolean)

        # R[j,t] residual life time of unit j at time t
        m.R = Var(self.units, self.TIMEs, domain=NonNegativeReals)

        # F[j,t] residual life restoration during maintenance
        m.F = Var(self.units, self.TIMEs, domain=NonNegativeReals)
        
        # B[i,j,k,t] size of batch assigned to task i in unit j at time t
        m.B = Var(self.tasks, self.units, self.opmodes, self.TIMEs, domain=NonNegativeReals)

        # A[i,j,t] total amount of material undergoing task i in unit j in planning time interval t
        m.A = Var(self.tasks, self.units, self.TIMESp, domain=NonNegativeReals)
        
        # S[s,t] inventory of state s at time t
        m.S = Var(self.states, self.TIMEs, domain=NonNegativeReals)
        
        # Q[j,t] inventory of unit j at time t
        m.Q = Var(self.units, self.TIMEs, domain=NonNegativeReals)

        # objectve
        m.Cost = Var(domain=NonNegativeReals)
        m.Value = Var(domain=NonNegativeReals)
        m.cons.add(m.Value == sum([self.price[s]*m.S[s,self.H] for s in self.states]))
        m.cons.add(m.Cost == sum([self.cost[(i,j)] * m.W[i,j,k,t] + self.vcost[(i,j)] * m.B[i,j,k,t]
                                   for i in self.tasks for j in self.K[i] for k in self.O[j] for t in self.TIMEs])) 
        m.Obj = Objective(expr = m.Value - m.Cost, sense = maximize)
       
        # scheduling horizon constraints 
        # unit constraints
        for j in self.units:
            rhs = 0
            for t in self.TIMEs:
                # a unit can only be allocated to one task 
                lhs = 0
                for i in self.I[j]:
                    for k in self.O[j]:
                        for tprime in self.TIMEs[(self.TIMEs <= t) & (self.TIMEs >= t-self.p[i,j,k]+1)]:
                            lhs += m.W[i,j,k,tprime]
                    for tprime in self.TIMEs[(self.TIMEs <= t) & (self.TIMEs >= t-self.tau[j]+1)]:
                        lhs += m.M[j,tprime]
                m.cons.add(lhs <= 1)
                
                # capacity constraints (see Konkili, Sec. 3.1.2)
                for i in self.I[j]:
                    for k in self.O[j]:
                    	m.cons.add(m.W[i,j,k,t]*self.Bmin[i,j] <= m.B[i,j,k,t])
                    	m.cons.add(m.B[i,j,k,t] <= m.W[i,j,k,t]*self.Bmax[i,j])
                    
                # unit mass balance
                rhs += sum([m.B[i,j,k,t] for i in self.I[j] for k in self.O[j]])
                for i in self.I[j]:
                    for s in self.S_[i]:
                        if t >= self.P[(i,s)]:
                            rhs -= self.rho_[(i,s)]*sum([m.B[i,j,k,max(self.TIMEs[self.TIMEs <= t-self.P[(i,s)]])] for k in self.O[j]])
                m.cons.add(m.Q[j,t] == rhs)
                rhs = m.Q[j,t]
                
                # switchover time constraints
                for (i1,i2) in self.changeoverTime.keys():
                    if (i1 in self.I[j]) and (i2 in self.I[j]):
                        for t1 in self.TIMEs[self.TIMEs <= (self.H - self.p[i1])]:
                            for t2 in self.TIMEs[(self.TIMEs >= t1 + self.p[i1])
                                            & (self.TIMEs < t1 + self.p[i1] + self.changeoverTime[(i1,i2)])]: 
                                m.cons.add(sum([m.W[i1,j,k,t1] for k in self.O[j]]) + sum([m.W[i2,j,k,t2] for k in self.O[k]]) <= 1)

                
                # terminal condition  
                m.cons.add(m.Q[j,self.H] == 0)

        # state constraints
        for s in self.states:
            rhs = self.init[s]
            for t in self.TIMEs:
                # state capacity constraint
                m.cons.add(m.S[s,t] <= self.C[s])
                # state mass balanace
                for i in self.T_[s]:
                    for j in self.K[i]:
                        for k in self.O[j]:
                            if t >= self.P[(i,s)] + self.p[i,j,k]: 
                                rhs += self.rho_[(i,s)]*m.B[i,j,k,max(self.TIMEs[self.TIMEs <= t-self.p[i,j,k]-self.P[(i,s)]])]             
                for i in self.T[s]:
                    for j in self.K[i]:
                        for k in self.O[j]:
                            rhs -= self.rho[(i,s)]*m.B[i,j,k,t]
                if (s,t) in self.Demand:
                    rhs -= self.Demand[s,t]
                m.cons.add(m.S[s,t] == rhs)
                rhs = m.S[s,t]

        # residual life constraints 
        for j in self.units:
            rhs = self.Rinit[j]
            for t in self.TIMEs:
                # constraints on F[j,t] and R[j,t]
                m.cons.add(m.F[j,t] <= self.Rmax[j]*m.M[j,t])
                m.cons.add(m.F[j,t] <= self.Rmax[j] - rhs)
                m.cons.add(m.F[j,t] >= self.Rmax[j]*m.M[j,t] - rhs)
                m.cons.add(0 <= m.R[j,t] <= self.Rmax[j])
                # residual life balance
                for i in self.I[j]:
                    for k in self.O[j]:
                        rhs -= self.D[i,j,k]*m.W[i,j,k,t]
                rhs += m.F[j,t]
                m.cons.add(m.R[j,t] == rhs)

        # planning horizon constraints
        # unit constraints
        for j in self.units:
            rhs = 0
            for t in self.TIMEp:
                # a unit can only be allocated to one task 
                lhs = 0
                for i in self.I[j]:
                    for k in self.O[j]:
                        lhs += m.W[i,j,k,t]*self.p[i,j,k]
                lhs += m.M[j,t]*self.tau[j]
                m.cons.add(lhs <= self.H)

                # capacity constraints (see Konkili, Sec. 3.1.2)
                for i in self.I[j]:
                    m.cons.add(sum([m.N[i,j,k,t] for k in self.O[j]])*self.Bmin[i,j] <= m.A[i,j,t])
                    m.cons.add(m.A[i,j,t] <= sum([m.N[i,j,k,t] for k in self.O[j]])*self.Bmax[i,j])
                
                # operating mode constraints
                for i in self.I[j]:
                    for k in self.O[j]:
                        m.cons.add(m.N[i,j,k,t] <= self.U*m.Mode[j,k,t])
                m.cons.add(sum([m.Mode[j,k,t] for k in self.O[j]]) == 1)

        # state constraints
        for s in self.states:
            rhs = self.init[s] # FIX!!
            for t in self.TIMEp:
                # state capacity constraint
                m.cons.add(m.S[s,t] <= self.C[s])
                # state mass balanace
                for i in self.T_[s]:
                    for j in self.K[i]:
                        rhs += self.rho_[(i,s)]*m.A[i,j,t]             
                for i in self.T[s]:
                    for j in self.K[i]:
                        rhs -= self.rho[(i,s)]*m.A[i,j,t]
                if (s,t) in self.Demand:
                    rhs -= self.Demand[s,t]
                m.cons.add(m.S[s,t] == rhs)
                rhs = m.S[s,t]

        # residual life constraints
        for j in self.units:
            rhs = self.Rinit[j] # FIX!!
            for t in self.TIMEp:
                # residual life balance
                for i in self.I[j]:
                    for k in self.O[j]:
                        rhs -= self.D[i,j,k]*m.N[i,j,k,t]
                rhs += m.F[j,t]
                m.cons.add(m.R[j,t] == rhs)
                # constraints on R and F
                m.cons.add(0 <= m.R[j,t] <= self.Rmax[j])


    def solve(self, solver='cplex'):
        self.solver = SolverFactory(solver)
#        self.solver.options['tmlim'] = 600
        self.solver.solve(self.model, tee=True).write()

    def gantt(self):
        model = self.model
        C = self.C
        H = self.H
        I = self.I
        p = self.p
        O = self.O

        gap = H/400
        idx = 1
        lbls = []
        ticks = []
        
        for s in self.states:
            plt.plot(self.TIMEs, [self.model.S[s,t]() for t in self.TIMEs])
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
                        if self.model.W[i,j,k,t]() > 0:
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
                        if model.W[i,j,k,t]() > 0:
                            plt.plot([t,t+p[i,j,k]], [idx,idx],'k', lw=24, alpha=0.5, solid_capstyle='butt')
                            plt.plot([t+gap,t+p[i,j,k]-gap], [idx,idx],'b', lw=20, solid_capstyle='butt')
                            txt = "{0:.2f}".format(model.B[i,j,k,t]())
                            col = {'Slow': 'green', 'Normal': 'yellow', 'Fast': 'red'}
                            plt.text(t+p[i]/2, idx, txt, color=col[k], weight='bold', ha='center', va='center')
        plt.xlim(0,self.H)
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
                    
        
