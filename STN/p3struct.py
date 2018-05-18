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
stn.unit('Heater',    'T1',    Bmin=0, Bmax=100, tm=18, rmax=110,
         # rinit=50, a=600, b=300)
         rinit=52, a=750, b=0)
stn.unit('Heater',    'T2',    Bmin=0, Bmax=100)
stn.unit('Reactor_1', 'T3', Bmin=0, Bmax=100, tm=15, rmax=120,
         # rinit=50, a=600, b=300)
         rinit=84, a=1150, b=0)
stn.unit('Reactor_1',    'T4',    Bmin=0, Bmax=100)
stn.unit('Reactor_1',    'T5',    Bmin=0, Bmax=100)
stn.unit('Reactor_2', 'T3', Bmin=0, Bmax=150, tm=15, rmax=100,
         # rinit=50, a=600, b=300)
         rinit=13, a=950, b=0)
stn.unit('Reactor_2',    'T4',    Bmin=0, Bmax=150)
stn.unit('Reactor_2',    'T5',    Bmin=0, Bmax=150)
stn.unit('Separator',    'T6',    Bmin=0, Bmax=300, tm=18, rmax=150,
         # rinit=50, a=600, b=300)
         rinit=105, a=800, b=0)
stn.unit('Separator',    'T7',    Bmin=0, Bmax=300)
stn.unit('Mixer_1',    'T7',    Bmin=20, Bmax=200, tm=9, rmax=90,
         # rinit=50, a=600, b=300)
         rinit=60, a=600, b=0)
stn.unit('Mixer_2',    'T7',    Bmin=20, Bmax=200, tm=9, rmax=90,
         # rinit=50, a=600, b=300)
         rinit=40, a=600, b=0)

stn.opmode('Normal')
stn.opmode('Fast')

stn.ijkdata('T1', 'Heater', 'Normal', 15, 5, 0.27*5)
stn.ijkdata('T1', 'Heater', 'Fast', 9, 8, 0.31*7)
stn.ijkdata('T2', 'Heater', 'Normal', 24, 4, 0.27*5)
stn.ijkdata('T2', 'Heater', 'Fast', 15, 7, 0.31*7)
stn.ijkdata('T3', 'Reactor_1', 'Normal', 30, 6, 0.27*6)
stn.ijkdata('T3', 'Reactor_1', 'Fast', 18, 9, 0.31*9)
stn.ijkdata('T4', 'Reactor_1', 'Normal', 15, 7, 0.27*6)
stn.ijkdata('T4', 'Reactor_1', 'Fast', 9, 11, 0.31*9)
stn.ijkdata('T5', 'Reactor_1', 'Normal', 30, 6, 0.27*6)
stn.ijkdata('T5', 'Reactor_1', 'Fast', 18, 10, 0.31*9)
stn.ijkdata('T3', 'Reactor_2', 'Normal', 30, 7, 0.27*6)
stn.ijkdata('T3', 'Reactor_2', 'Fast', 18, 11, 0.31*9)
stn.ijkdata('T4', 'Reactor_2', 'Normal', 15, 6, 0.27*6)
stn.ijkdata('T4', 'Reactor_2', 'Fast', 9, 9, 0.31*9)
stn.ijkdata('T5', 'Reactor_2', 'Normal', 30, 6, 0.27*6)
stn.ijkdata('T5', 'Reactor_2', 'Fast', 18, 10, 0.31*9)
stn.ijkdata('T6', 'Separator', 'Normal', 45, 4, 0.22*4)
stn.ijkdata('T6', 'Separator', 'Fast', 27, 6, 0.27*6)
stn.ijkdata('T7', 'Separator', 'Normal', 12, 4, 0.22*4)
stn.ijkdata('T7', 'Separator', 'Fast', 6, 7, 0.27*6)
stn.ijkdata('T7', 'Mixer_1', 'Normal', 30, 4, 0.22*4)
stn.ijkdata('T7', 'Mixer_1', 'Fast', 18, 6, 0.27*6)
stn.ijkdata('T7', 'Mixer_2', 'Normal', 30, 4, 0.22*4)
stn.ijkdata('T7', 'Mixer_2', 'Fast', 18, 6, 0.27*6)

with open("../data/p3.dat", "wb") as dill_file:
    dill.dump(stn, dill_file)
