#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import dill
sys.path.append('../STN/modules')
from stn import stnStruct # noqa

# create instance
stn = stnStruct()

# states
stn.state('S1',     init=2000000)
stn.state('S2',     init=2000000)
stn.state('S3',     init=0, scost=1)
stn.state('S4',     init=0, scost=1)
stn.state('S5',     init=0, scost=1)
stn.state('S6',     init=50, scost=1)
stn.state('S7',     init=50, scost=1)
stn.state('S8',     init=2000000)
stn.state('S9',     init=0, scost=1)
stn.state('S10',     init=0, scost=1)
stn.state('S11',     init=2000000)
stn.state('S12',     init=0, scost=5, prod=True)
stn.state('S13',     init=0, scost=5, prod=True)

# state to task arcs
stn.stArc('S1',   'T1')
stn.stArc('S2',   'T3')
stn.stArc('S3',   'T4', rho=0.5)
stn.stArc('S4',   'T4', rho=0.5)
stn.stArc('S5',   'T6')
stn.stArc('S6',   'T2', rho=0.25)
stn.stArc('S7',   'T7', rho=0.4)
stn.stArc('S8',   'T2', rho=0.75)
stn.stArc('S9',   'T5')
stn.stArc('S10',   'T7', rho=0.4)
stn.stArc('S11',   'T7', rho=0.2)

# task to state arcs
stn.tsArc('T1', 'S3', rho=1.0)
stn.tsArc('T2', 'S9', rho=1.0)
stn.tsArc('T3', 'S4', rho=1.0)
stn.tsArc('T4', 'S5', rho=1.0)
stn.tsArc('T5', 'S13', rho=0.4)
stn.tsArc('T5', 'S10', rho=0.6)
stn.tsArc('T6', 'S4', rho=0.1)
stn.tsArc('T6', 'S6', rho=0.4)
stn.tsArc('T6', 'S7', rho=0.5)
stn.tsArc('T7', 'S12', rho=1.0)

# unit-task data
stn.unit('Heater',    'T1',    Bmin=0, Bmax=100, tm=20, rmax=120,
         # rinit=50, a=600, b=300)
         rinit=10, a=600, b=0)
stn.unit('Heater',    'T2',    Bmin=0, Bmax=100)
stn.unit('Reactor_1', 'T3', Bmin=0, Bmax=100, tm=15, rmax=100,
         # rinit=50, a=600, b=300)
         rinit=70, a=600, b=0)
stn.unit('Reactor_1',    'T4',    Bmin=0, Bmax=100)
stn.unit('Reactor_1',    'T5',    Bmin=0, Bmax=100)
stn.unit('Reactor_2', 'T3', Bmin=0, Bmax=150, tm=15, rmax=100,
         # rinit=50, a=600, b=300)
         rinit=70, a=600, b=0)
stn.unit('Reactor_2',    'T4',    Bmin=0, Bmax=150)
stn.unit('Reactor_2',    'T5',    Bmin=0, Bmax=150)
stn.unit('Separator',    'T6',    Bmin=0, Bmax=300, tm=18, rmax=150,
         # rinit=50, a=600, b=300)
         rinit=45, a=500, b=0)
stn.unit('Separator',    'T7',    Bmin=0, Bmax=300)
stn.unit('Mixer_1',    'T7',    Bmin=20, Bmax=200, tm=9, rmax=90,
         # rinit=50, a=600, b=300)
         rinit=60, a=400, b=0)
stn.unit('Mixer_2',    'T7',    Bmin=20, Bmax=200, tm=9, rmax=90,
         # rinit=50, a=600, b=300)
         rinit=60, a=400, b=0)

stn.opmode('Normal')
stn.opmode('Fast')

stn.ijkdata('T1', 'Heater', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T1', 'Heater', 'Fast', 15, 7, 0.31*7)
stn.ijkdata('T2', 'Heater', 'Normal', 39, 5, 0.27*5)
stn.ijkdata('T2', 'Heater', 'Fast', 24, 7, 0.31*7)
stn.ijkdata('T3', 'Reactor_1', 'Normal', 51, 6, 0.27*6)
stn.ijkdata('T3', 'Reactor_1', 'Fast', 33, 9, 0.31*9)
stn.ijkdata('T4', 'Reactor_1', 'Normal', 24, 6, 0.27*6)
stn.ijkdata('T4', 'Reactor_1', 'Fast', 15, 9, 0.31*9)
stn.ijkdata('T5', 'Reactor_1', 'Normal', 51, 6, 0.27*6)
stn.ijkdata('T5', 'Reactor_1', 'Fast', 33, 9, 0.31*9)
stn.ijkdata('T3', 'Reactor_2', 'Normal', 51, 6, 0.27*6)
stn.ijkdata('T3', 'Reactor_2', 'Fast', 33, 9, 0.31*9)
stn.ijkdata('T4', 'Reactor_2', 'Normal', 24, 6, 0.27*6)
stn.ijkdata('T4', 'Reactor_2', 'Fast', 15, 9, 0.31*9)
stn.ijkdata('T5', 'Reactor_2', 'Normal', 51, 6, 0.27*6)
stn.ijkdata('T5', 'Reactor_2', 'Fast', 33, 9, 0.31*9)
stn.ijkdata('T6', 'Separator', 'Normal', 78, 4, 0.22*4)
stn.ijkdata('T6', 'Separator', 'Fast', 51, 6, 0.27*6)
stn.ijkdata('T7', 'Separator', 'Normal', 18, 4, 0.22*4)
stn.ijkdata('T7', 'Separator', 'Fast', 12, 6, 0.27*6)
stn.ijkdata('T7', 'Mixer_1', 'Normal', 51, 4, 0.22*4)
stn.ijkdata('T7', 'Mixer_1', 'Fast', 33, 6, 0.27*6)
stn.ijkdata('T7', 'Mixer_2', 'Normal', 51, 4, 0.22*4)
stn.ijkdata('T7', 'Mixer_2', 'Fast', 33, 6, 0.27*6)

with open("../data/p3.dat", "wb") as dill_file:
    dill.dump(stn, dill_file)
