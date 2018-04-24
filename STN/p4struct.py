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
stn.state('F1',     init=2000000)
stn.state('F2',     init=2000000)
stn.state('I1',     init=0, capacity=200, scost=1)
stn.state('I2',     init=0, capacity=100, scost=1)
stn.state('I3',     init=0, capacity=500, scost=1)
stn.state('P1',     init=0, capacity=1000, scost=5)
stn.state('P2',     init=0, capacity=1000, scost=5)

# state to task arcs
stn.stArc('F1',   'T1', rho=0.8)
stn.stArc('F2',   'T2')
stn.stArc('I1',   'T1', rho=0.2)
stn.stArc('I2',   'T4', rho=0.6)
stn.stArc('I3',   'T3')
stn.stArc('I3',   'T4', rho=0.4)

# task to state arcs
stn.tsArc('T1', 'I3', rho=1.0)
stn.tsArc('T2', 'I1', rho=0.3)
stn.tsArc('T2', 'I2', rho=0.7)
stn.tsArc('T3', 'P1', rho=1.0)
stn.tsArc('T4', 'P2', rho=1.0)

# unit-task data
stn.unit('R1',    'T1',    Bmin=0, Bmax=100, tm=21, rmax=120,
         # rinit=50, a=600, b=300)
         rinit=10, a=600, b=0)
stn.unit('R1',    'T2',    Bmin=0, Bmax=100)
stn.unit('R2',    'T1',    Bmin=0, Bmax=100, tm=21, rmax=120,
         # rinit=50, a=600, b=300)
         rinit=10, a=600, b=0)
stn.unit('R2',    'T2',    Bmin=0, Bmax=100)
stn.unit('R3',    'T3',    Bmin=0, Bmax=100, tm=21, rmax=120,
         # rinit=50, a=600, b=300)
         rinit=10, a=600, b=0)
stn.unit('R3',    'T4',    Bmin=0, Bmax=100)

stn.opmode('Slow')
stn.opmode('Normal')
stn.opmode('Fast')

stn.ijkdata('T1', 'R1', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T1', 'R1', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T1', 'R1', 'Fast', 18, 8, 0.27*8)
stn.ijkdata('T2', 'R1', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T2', 'R1', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T2', 'R1', 'Fast', 18, 8, 0.27*8)
stn.ijkdata('T1', 'R2', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T1', 'R2', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T1', 'R2', 'Fast', 18, 8, 0.27*8)
stn.ijkdata('T2', 'R2', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T2', 'R2', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T2', 'R2', 'Fast', 18, 8, 0.27*8)
stn.ijkdata('T3', 'R3', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T3', 'R3', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T3', 'R3', 'Fast', 18, 8, 0.27*8)
stn.ijkdata('T4', 'R3', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T4', 'R3', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T4', 'R3', 'Fast', 18, 8, 0.27*8)

import ipdb; ipdb.set_trace()
with open("p4D.dat", "wb") as dill_file:
    dill.dump(stn, dill_file)
