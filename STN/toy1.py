#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Toy example of STN with degradation.

Units:
    Heater

Tasks:
    Heating

Operating modes:
    Slow
    Normal

"""

import sys
sys.path.append('../STN/modules')

from stn_deg_det import stnModel, stnModelRobust # noqa

# create instance
model = stnModel()
stn = model.stn

# states
stn.state('F',     init=2000000)   # Feed F
stn.state('P', price=10, scost=5)  # Product P

# state to task arcs
stn.stArc('F',   'Heating')

# task to state arcs
stn.tsArc('Heating',    'P',      rho=1.0)

# unit-task data
stn.unit('Heater',    'Heating',    Bmin=40, Bmax=100, tm=15, rmax=80,
         rinit=10, a=600, b=300)

# operating mode and degradation data
stn.opmode('Slow')
stn.opmode('Normal')
stn.ijkdata('Heating', 'Heater', 'Slow', 9, 5)
stn.ijkdata('Heating', 'Heater', 'Normal', 6, 10)

# time horizons
Ts = 30
dTs = 3
Tp = 30*4
dTp = 30
TIMEs = range(0, Ts, dTs)
TIMEp = range(Ts, Tp, dTp)

# demand for Product P
demand_1 = [150, 88, 125, 67, 166]
model.demand('P', Ts-dTs, demand_1[0])

for i in range(0, len(TIMEp)):
    model.demand('P', TIMEp[i], demand_1[i+1])

# build and solve model
model.build(TIMEs, TIMEp)
model.solve('cplex')

# get results
model.gantt()
model.trace()
model.trace_planning()
