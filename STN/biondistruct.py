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
stn.state('FeedA',     init=2000000)
stn.state('FeedB',     init=2000000)
stn.state('FeedC',     init=2000000)
stn.state('HotA',      price=-1, capacity=100, scost=1)
stn.state('IntAB',     price=-1, capacity=200, scost=1)
stn.state('IntBC',     price=-1, capacity=150, scost=1)
stn.state('ImpureE',   price=-1, capacity=100, scost=1)
stn.state('Product_1', price=10, scost=5)
stn.state('Product_2', price=10, scost=5)

# state to task arcs
stn.stArc('FeedA',   'Heating')
stn.stArc('FeedB',   'Reaction_1', rho=0.5)
stn.stArc('FeedC',   'Reaction_1', rho=0.5)
stn.stArc('FeedC',   'Reaction_3', rho=0.2)
stn.stArc('HotA',    'Reaction_2', rho=0.4)
stn.stArc('IntAB',   'Reaction_3', rho=0.8)
stn.stArc('IntBC',   'Reaction_2', rho=0.6)
stn.stArc('ImpureE', 'Separation')

# task to state arcs
stn.tsArc('Heating',    'HotA',      rho=1.0)
stn.tsArc('Reaction_2', 'IntAB',     rho=0.6)
stn.tsArc('Reaction_2', 'Product_1', rho=0.4)
stn.tsArc('Reaction_1', 'IntBC')
stn.tsArc('Reaction_3', 'ImpureE')
stn.tsArc('Separation', 'IntAB',     rho=0.1)
stn.tsArc('Separation', 'Product_2', rho=0.9)

# unit-task data
stn.unit('Heater',    'Heating',    Bmin=40, Bmax=100, tm=15, rmax=80,
         rinit=50, a=600, b=300)
stn.unit('Reactor_1', 'Reaction_1', Bmin=32, Bmax=80, tm=21, rmax=150,
         rinit=100, a=1500, b=600)
stn.unit('Reactor_1', 'Reaction_2', Bmin=32, Bmax=80, tm=21)
stn.unit('Reactor_1', 'Reaction_3', Bmin=32, Bmax=80, tm=21)
stn.unit('Reactor_2', 'Reaction_1', Bmin=20, Bmax=50, tm=24, rmax=160,
         rinit=40, a=2500, b=500)
stn.unit('Reactor_2', 'Reaction_2', Bmin=20, Bmax=50, tm=24)
stn.unit('Reactor_2', 'Reaction_3', Bmin=20, Bmax=50, tm=24)
stn.unit('Still',     'Separation', Bmin=80, Bmax=200, tm=15, rmax=100,
         rinit=60, a=2700, b=1500)

stn.opmode('Slow')
stn.opmode('Normal')
stn.opmode('Fast')

stn.ijkdata('Heating', 'Heater', 'Slow', 9, 1)
stn.ijkdata('Heating', 'Heater', 'Normal', 6, 2)
stn.ijkdata('Heating', 'Heater', 'Fast', 3, 3)
stn.ijkdata('Reaction_1', 'Reactor_1', 'Slow', 27, 4)
stn.ijkdata('Reaction_1', 'Reactor_1', 'Normal', 15, 5)
stn.ijkdata('Reaction_1', 'Reactor_1', 'Fast', 9, 8)
stn.ijkdata('Reaction_1', 'Reactor_2', 'Slow', 30, 4)
stn.ijkdata('Reaction_1', 'Reactor_2', 'Normal', 18, 5)
stn.ijkdata('Reaction_1', 'Reactor_2', 'Fast', 12, 10)
stn.ijkdata('Reaction_2', 'Reactor_1', 'Slow', 36, 1)
stn.ijkdata('Reaction_2', 'Reactor_1', 'Normal', 21, 3)
stn.ijkdata('Reaction_2', 'Reactor_1', 'Fast', 15, 5)
stn.ijkdata('Reaction_2', 'Reactor_2', 'Slow', 33, 2)
stn.ijkdata('Reaction_2', 'Reactor_2', 'Normal', 18, 4)
stn.ijkdata('Reaction_2', 'Reactor_2', 'Fast', 12, 4)
stn.ijkdata('Reaction_3', 'Reactor_1', 'Slow', 30, 3)
stn.ijkdata('Reaction_3', 'Reactor_1', 'Normal', 18, 7)
stn.ijkdata('Reaction_3', 'Reactor_1', 'Fast', 6, 8)
stn.ijkdata('Reaction_3', 'Reactor_2', 'Slow', 24, 2)
stn.ijkdata('Reaction_3', 'Reactor_2', 'Normal', 21, 5)
stn.ijkdata('Reaction_3', 'Reactor_2', 'Fast', 12, 9)
stn.ijkdata('Separation', 'Still', 'Slow', 15, 2)
stn.ijkdata('Separation', 'Still', 'Normal', 9, 5)
stn.ijkdata('Separation', 'Still', 'Fast', 6, 6)

with open("biondiRstruct.dat", "wb") as dill_file:
    dill.dump(stn, dill_file)
