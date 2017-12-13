#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
sys.path.append('../STN')
from STNdegBlocks import STN
import numpy as np

def pertD(mean, sd):
    return max(np.random.normal(mean,sd),0)
    #return mean
# create instance
N = 10
for n in range(0,N):
    stn = STN()

    # states
    stn.state('FeedA',     init = 2000000)
    stn.state('FeedB',     init = 2000000)
    stn.state('FeedC',     init = 2000000)
    stn.state('HotA',      price = -1, capacity = 100, scost = 1)
    stn.state('IntAB',     price = -1, capacity = 200, scost = 1)
    stn.state('IntBC',     price = -1, capacity = 150, scost = 1)
    stn.state('ImpureE',   price = -1, capacity = 100, scost = 1)
    stn.state('Product_1', price = 10, scost = 5)
    stn.state('Product_2', price = 10, scost = 5)

    # state to task arcs
    stn.stArc('FeedA',   'Heating')
    stn.stArc('FeedB',   'Reaction_1', rho = 0.5)
    stn.stArc('FeedC',   'Reaction_5', rho = 0.5)
    stn.stArc('FeedC',   'Reaction_3', rho = 0.2)
    stn.stArc('HotA',    'Reaction_2', rho = 0.4)
    stn.stArc('IntAB',   'Reaction_3', rho = 0.8)
    stn.stArc('IntBC',   'Reaction_2', rho = 0.6)
    stn.stArc('ImpureE', 'Separation')

    # task to state arcs
    stn.tsArc('Heating',    'HotA',      rho = 1.0)
    stn.tsArc('Reaction_2', 'IntAB',     rho = 0.6)
    stn.tsArc('Reaction_2', 'Product_1', rho = 0.4)
    stn.tsArc('Reaction_1', 'IntBC')
    stn.tsArc('Reaction_3', 'ImpureE')
    stn.tsArc('Separation', 'IntAB',     rho = 0.1)
    stn.tsArc('Separation', 'Product_2', rho = 0.9)

    # unit-task data
    stn.unit('Heater',    'Heating',    Bmin = 40, Bmax = 100, tm = 15, rmax = 80,
             rinit = 30, a = 600, b =300)
    stn.unit('Reactor_1', 'Reaction_1', Bmin = 32, Bmax =  80, tm = 21, rmax = 150,
             rinit = 50, a = 1500, b =600)
    stn.unit('Reactor_1', 'Reaction_2', Bmin = 32, Bmax =  80, tm = 21)
    stn.unit('Reactor_1', 'Reaction_3', Bmin = 32, Bmax =  80, tm = 21)
    stn.unit('Reactor_2', 'Reaction_1', Bmin = 20, Bmax =  50, tm = 24, rmax = 160,
             rinit = 120, a = 2500, b =500)
    stn.unit('Reactor_2', 'Reaction_2', Bmin = 20, Bmax =  50, tm = 24)
    stn.unit('Reactor_2', 'Reaction_3', Bmin = 20, Bmax =  50, tm = 24)
    stn.unit('Still',     'Separation', Bmin = 80, Bmax = 200, tm = 15, rmax = 100,
             rinit = 40, a = 2700, b = 1500)

    stn.opmode('Slow')
    stn.opmode('Normal')
    stn.opmode('Fast')

    stn.ijkdata('Heating', 'Heater', 'Slow', 9, pertD(1,0.1))
    stn.ijkdata('Heating', 'Heater', 'Normal', 6, pertD(2,0.1))
    stn.ijkdata('Heating', 'Heater', 'Fast', 3, pertD(3,0.1))
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Slow', 27, pertD(4,0.1))
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Normal', 15, pertD(5,0.1))
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Fast', 9, pertD(8,0.1))
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Slow', 30, pertD(4,0.1))
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Normal', 18, pertD(5,0.1))
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Fast', 12, pertD(10,0.1))
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Slow', 36, pertD(1,0.1))
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Normal', 21, pertD(3,0.1))
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Fast', 15, pertD(5,0.1))
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Slow', 33, pertD(2,0.1))
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Normal', 18, pertD(4,0.1))
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Fast', 12, pertD(4,0.1))
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Slow', 30, pertD(3,0.1))
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Normal', 18, pertD(7,0.1))
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Fast', 6, pertD(8,0.1))
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Slow', 24, pertD(2,0.1))
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Normal', 21, pertD(5,0.1))
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Fast', 12, pertD(9,0.1))
    stn.ijkdata('Separation', 'Still', 'Slow', 15, pertD(2,0.1))
    stn.ijkdata('Separation', 'Still', 'Normal', 9, pertD(5,0.1))
    stn.ijkdata('Separation', 'Still', 'Fast', 6, pertD(6,0.1))

    demand_1 = [150, 88, 125, 67, 166, 203, 90, 224, 174, 126, 66, 119, 234, 64,
                103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
    demand_2 = [200, 150, 197, 296, 191, 193, 214, 294, 247, 313, 226, 121, 197,
                242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]

    Ts = 168
    dTs = 3
    Tp = 168*24
    dTp = 168
    TIMEs = range(0,Ts,dTs)
    TIMEp = range(Ts,Tp,dTp)

    stn.demand('Product_1', Ts-dTs, demand_1[0])
    stn.demand('Product_2', Ts-dTs, demand_2[0])

    for i in range(0,len(TIMEp)):
        stn.demand('Product_1', TIMEp[i], demand_1[i+1])
        stn.demand('Product_2', TIMEp[i], demand_2[i+1])

    stn.build(TIMEs,TIMEp)
    stn.solve('cplex',prefix=str(n))
    stn.gantt(prefix=str(n))
    stn.trace(prefix=str(n))
    stn.trace_planning(prefix=str(n))
    stn.eval()
