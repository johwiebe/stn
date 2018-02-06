#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Toy example of STN with degradation.

Units:
    Heater
    Reactor

Tasks:
    Heating
    Reaction_1
    Reaction_2

Operating modes:
    Slow
    Normal

"""

import sys
sys.path.append('../STN/modules')

from stn import stnModelRobust # noqa
import deg # noqa

# create instance
model = stnModelRobust()
stn = model.stn

# states
stn.state('F1',     init=2000000)   # Feed F1
stn.state('F2',     init=2000000)   # Feed F2
stn.state('P1', price=10, scost=9)  # Product P1
stn.state('P2', price=20, scost=5)  # Product P2
stn.state('I1', scost=15)  # Product P2

# state to task arcs
stn.stArc('F1',   'Heating')
stn.stArc('F2',   'Reaction_1', rho=0.6)
stn.stArc('I1', 'Reaction_1', rho=0.4)
stn.stArc('P1', 'Reaction_2', rho=0.3)
stn.stArc('I1', 'Reaction_2', rho=0.7)


# task to state arcs
stn.tsArc('Heating',    'I1')
stn.tsArc('Reaction_1',    'P1')
stn.tsArc('Reaction_2',    'P2')

# unit-task data
stn.unit('Heater', 'Heating', Bmin=40, Bmax=100, tm=5, rmax=80,
         rinit=43, a=600, b=300)
stn.unit('Reactor', 'Reaction_1', Bmin=30, Bmax=140, tm=5, rmax=120,
         rinit=70, a=600, b=300)
stn.unit('Reactor', 'Reaction_2', Bmin=30, Bmax=140, tm=5, rmax=120,
         rinit=70, a=600, b=300)

# operating mode and degradation data
stn.opmode('Slow')
stn.opmode('Normal')
stn.ijkdata('Heating', 'Heater', 'Slow', 9, 5)
stn.ijkdata('Heating', 'Heater', 'Normal', 6, 11)
stn.ijkdata('Reaction_1', 'Reactor', 'Slow', 6, 7)
stn.ijkdata('Reaction_1', 'Reactor', 'Normal', 4, 9)
stn.ijkdata('Reaction_2', 'Reactor', 'Slow', 10, 5)
stn.ijkdata('Reaction_2', 'Reactor', 'Normal', 6, 13)

# time horizons
Ts = 30
dTs = 1
Tp = 30*8
dTp = 30
TIMEs = range(0, Ts, dTs)
TIMEp = range(0, Tp, dTp)

# demand for Product P
# demand_1 = [150, 88, 125, 67, 166]
demand_1 = [150, 88, 125, 67, 166, 125, 302, 94, 300, 300, 300, 300]
demand_2 = [50, 188, 131, 27, 241, 155, 122, 104, 300, 300, 300, 300]
# model.demand('P1', Ts-dTs, demand_1[0])
# model.demand('P2', Ts-dTs, demand_2[0])

for i in range(0, len(TIMEp)):
    model.demand('P1', TIMEp[i], demand_1[i])
    model.demand('P2', TIMEp[i], demand_2[i])
# build and solve model
# model.build(TIMEs, TIMEp, objective="terminal", decisionrule="integer")
# model.build([Ts, dTs, Tp, dTp], objective="terminal",
#             decisionrule="continuous")
model.solve([Ts, dTs, Tp, dTp],
            solver="cplex",
            objective="terminal",
            decisionrule="continuous",
            periods=5,
            prefix="toy2R",
            rdir="/home/jw3617/STN/results")

import ipdb; ipdb.set_trace()  # noqa
# S = deg.simulate_wiener(model, "Heater", Sinit=43, N=300)
# Ninfeasible = deg.simulate_wiener(model, "Reactor", Sinit=70, N=300)
# print(Ninfeasible)
