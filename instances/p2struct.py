#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import os
import dill
os.chdir(os.getcwd() + "/" + os.path.split(sys.argv[0])[0])
sys.path.append('..')
from stn import stnStruct # noqa

# create instance
stn = stnStruct()

# states
stn.state('S1',     init=2000000)
stn.state('S2',     init=0, scost=1)
stn.state('S3',     init=0, scost=1)
stn.state('S4',     init=0, scost=1, prod=True)

# state to task arcs
stn.stArc('S1',   'T1')
stn.stArc('S2',   'T2')
stn.stArc('S3',   'T3')

# task to state arcs
stn.tsArc('T1', 'S2', rho=1.0)
stn.tsArc('T2', 'S3', rho=1.0)
stn.tsArc('T3', 'S4', rho=1.0)

# unit-task data
stn.unit('U1',    'T1',    Bmin=0, Bmax=100, tm=21, rmax=120,
         # rinit=50, a=600, b=300)
         rinit=10, a=600, b=0)
stn.unit('U2', 'T1', Bmin=0, Bmax=150, tm=15, rmax=100,
         # rinit=50, a=600, b=300)
         rinit=70, a=600, b=0)
stn.unit('U3',    'T2',    Bmin=0, Bmax=200, tm=18, rmax=150,
         # rinit=50, a=600, b=300)
         rinit=45, a=500, b=0)
stn.unit('U4',    'T3',    Bmin=0, Bmax=150, tm=9, rmax=90,
         # rinit=50, a=600, b=300)
         rinit=60, a=400, b=0)
stn.unit('U5',    'T3',    Bmin=0, Bmax=150, tm=13, rmax=80,
         # rinit=50, a=600, b=300)
         rinit=30, a=400, b=0)

stn.opmode('Slow')
stn.opmode('Normal')
stn.opmode('Fast')

stn.ijkdata('T1', 'U1', 'Slow', 33, 3, 0.22*3)
stn.ijkdata('T1', 'U1', 'Normal', 24, 5, 0.27*5)
stn.ijkdata('T1', 'U1', 'Fast', 15, 7, 0.31*7)
stn.ijkdata('T1', 'U2', 'Slow', 30, 3, 0.22*3)
stn.ijkdata('T1', 'U2', 'Normal', 21, 6, 0.27*6)
stn.ijkdata('T1', 'U2', 'Fast', 12, 9, 0.31*9)
stn.ijkdata('T2', 'U3', 'Slow', 24, 4, 0.22*4)
stn.ijkdata('T2', 'U3', 'Normal', 18, 6, 0.27*6)
stn.ijkdata('T2', 'U3', 'Fast', 12, 10, 0.31*10)
stn.ijkdata('T3', 'U4', 'Slow', 21, 2, 0.22*2)
stn.ijkdata('T3', 'U4', 'Normal', 15, 4, 0.27*4)
stn.ijkdata('T3', 'U4', 'Fast', 9, 7, 0.31*7)
stn.ijkdata('T3', 'U5', 'Slow', 18, 2, 0.22*2)
stn.ijkdata('T3', 'U5', 'Normal', 12, 4, 0.27*4)
stn.ijkdata('T3', 'U5', 'Fast', 6, 6, 0.31*6)

with open("../data/p2.dat", "wb") as dill_file:
    dill.dump(stn, dill_file)
