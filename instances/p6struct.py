#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
stn.state('F1',     init=2000000)
stn.state('F2',     init=2000000)
stn.state('F3',     init=2000000)
stn.state('I4',     init=0, capacity=1000, scost=1)
stn.state('I5',     init=0, capacity=1000, scost=1)
stn.state('I6',     init=0, capacity=1500, scost=1)
stn.state('I7',     init=0, capacity=2000, scost=1)
stn.state('I8',     init=0, capacity=1000, scost=1)
stn.state('I9',     init=0, capacity=4000, scost=1)
stn.state('P1',     init=0, scost=5, prod=True)
stn.state('P2',     init=0, scost=5, prod=True)
stn.state('P3',     init=0, scost=5, prod=True)
stn.state('P4',     init=0, scost=5, prod=True)

# state to task arcs
stn.stArc('F1',   'T1')
stn.stArc('F2',   'T2')
stn.stArc('F3',   'T3', rho=0.6)
stn.stArc('I4',   'T4', rho=0.5)
stn.stArc('I5',   'T4', rho=0.5)
stn.stArc('I5',   'T3', rho=0.4)
stn.stArc('I6',   'T6')
stn.stArc('I6',   'T7')
stn.stArc('I7',   'T8', rho=0.5)
stn.stArc('I8',   'T8', rho=0.5)
stn.stArc('I9',   'T5')

# task to state arcs
stn.tsArc('T1', 'I4', rho=1.0)
stn.tsArc('T2', 'I5', rho=1.0)
stn.tsArc('T3', 'I6', rho=1.0)
stn.tsArc('T4', 'P1', rho=0.3)
stn.tsArc('T4', 'I9', rho=0.7)
stn.tsArc('T5', 'P2', rho=1.0)
stn.tsArc('T6', 'I8', rho=1.0)
stn.tsArc('T7', 'P3', rho=0.3)
stn.tsArc('T7', 'I7', rho=0.7)
stn.tsArc('T8', 'P4', rho=1.0)

# unit-task data
stn.unit('R1',    'T1',    Bmin=0, Bmax=1000, tm=21, rmax=100,
         rinit=77, a=1000, b=0)
stn.unit('R2',    'T3',    Bmin=0, Bmax=2500, tm=21, rmax=100,
         rinit=80, a=1700, b=0)
stn.unit('R2',    'T7',    Bmin=0, Bmax=2500)
stn.unit('R3',    'T4',    Bmin=0, Bmax=3500, tm=21, rmax=100,
         rinit=90, a=2000, b=0)
stn.unit('R4',    'T2',    Bmin=0, Bmax=1500, tm=21, rmax=100,
         rinit=17, a=1200, b=0)
stn.unit('R5',    'T6',    Bmin=0, Bmax=1000, tm=21, rmax=100,
         rinit=40, a=1000, b=0)
stn.unit('R6',    'T5',    Bmin=0, Bmax=4000, tm=21, rmax=100,
         rinit=33, a=2100, b=0)
stn.unit('R6',    'T8',    Bmin=0, Bmax=4000)

stn.opmode('Slow')
stn.opmode('Normal')

stn.ijkdata('T1', 'R1', 'Slow', 18, 3, 0.22*3)
stn.ijkdata('T1', 'R1', 'Normal', 12, 5, 0.27*5)
stn.ijkdata('T3', 'R2', 'Slow', 33, 5, 0.22*3)
stn.ijkdata('T3', 'R2', 'Normal', 21, 8, 0.27*5)
stn.ijkdata('T7', 'R2', 'Slow', 33, 6, 0.22*3)
stn.ijkdata('T7', 'R2', 'Normal', 21, 9, 0.27*5)
stn.ijkdata('T4', 'R3', 'Slow', 42, 5, 0.22*3)
stn.ijkdata('T4', 'R3', 'Normal', 21, 11, 0.27*5)
stn.ijkdata('T2', 'R4', 'Slow', 24, 3, 0.22*3)
stn.ijkdata('T2', 'R4', 'Normal', 15, 7, 0.27*5)
stn.ijkdata('T6', 'R5', 'Slow', 18, 3, 0.22*3)
stn.ijkdata('T6', 'R5', 'Normal', 12, 5, 0.27*5)
stn.ijkdata('T5', 'R6', 'Slow', 45, 7, 0.22*3)
stn.ijkdata('T5', 'R6', 'Normal', 30, 10, 0.27*5)
stn.ijkdata('T8', 'R6', 'Slow', 45, 6, 0.22*3)
stn.ijkdata('T8', 'R6', 'Normal', 30, 10, 0.27*5)

with open("../data/p6.dat", "wb") as dill_file:
    dill.dump(stn, dill_file)
