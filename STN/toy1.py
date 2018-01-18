#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
sys.path.append('../STN/modules')

from STNDeterministic import STN # noqa

# from robustGounaris import STN

# create instance
stn = STN()

# states
stn.state('F',     init=2000000)
stn.state('P', price=10, scost=5)

# state to task arcs
stn.stArc('F',   'Heating')

# task to state arcs
stn.tsArc('Heating',    'P',      rho=1.0)

# unit-task data
stn.unit('Heater',    'Heating',    Bmin=40, Bmax=100, tm=15, rmax=80,
         rinit=4, a=600, b=300)

stn.opmode('Slow')
stn.opmode('Normal')

stn.ijkdata('Heating', 'Heater', 'Slow', 9, 5)
stn.ijkdata('Heating', 'Heater', 'Normal', 6, 10)

demand_1 = [150, 88, 125, 67, 166]

Ts = 30
dTs = 3
Tp = 30*4
dTp = 30
TIMEs = range(0, Ts, dTs)
TIMEp = range(Ts, Tp, dTp)

stn.demand('P', Ts-dTs, demand_1[0])

for i in range(0, len(TIMEp)):
    stn.demand('P', TIMEp[i], demand_1[i+1])

stn.build(TIMEs, TIMEp)
stn.solve('cplex')
stn.gantt()
stn.trace()
stn.trace_planning()
import ipdb; ipdb.set_trace() # noqa
