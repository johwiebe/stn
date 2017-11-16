#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
sys.path.append('../STN')
from STNdeg import STN

# create instance
stn = STN()

# states
stn.state('FeedA',     init = 2000000)
stn.state('FeedB',     init = 2000000)
stn.state('FeedC',     init = 2000000)
stn.state('HotA',      price = -1, capacity = 100)
stn.state('IntAB',     price = -1, capacity = 200)
stn.state('IntBC',     price = -1, capacity = 150)
stn.state('ImpureE',   price = -1, capacity = 100)
stn.state('Product_1', price = 10)
stn.state('Product_2', price = 10)

# state to task arcs
stn.stArc('FeedA',   'Heating')
stn.stArc('FeedB',   'Reaction_1', rho = 0.5)
stn.stArc('FeedC',   'Reaction_1', rho = 0.5)
stn.stArc('FeedC',   'Reaction_3', rho = 0.2)
stn.stArc('HotA',    'Reaction_2', rho = 0.4)
stn.stArc('IntAB',   'Reaction_3', rho = 0.8)
stn.stArc('IntBC',   'Reaction_2', rho = 0.6)
stn.stArc('ImpureE', 'Separation')

# task to state arcs
stn.tsArc('Heating',    'HotA',      rho = 1.0, dur = 1)
stn.tsArc('Reaction_2', 'IntAB',     rho = 0.6, dur = 2)
stn.tsArc('Reaction_2', 'Product_1', rho = 0.4, dur = 2)
stn.tsArc('Reaction_1', 'IntBC',     dur = 2)
stn.tsArc('Reaction_3', 'ImpureE',   dur = 1)
stn.tsArc('Separation', 'IntAB',     rho = 0.1, dur = 2)
stn.tsArc('Separation', 'Product_2', rho = 0.9, dur = 1)

# unit-task data
stn.unit('Heater',    'Heating',    Bmin = 0, Bmax = 100, tm = 15, rmax = 80, rinit = 30)
stn.unit('Reactor_1', 'Reaction_1', Bmin = 0, Bmax =  80, tm = 21, rmax = 150, rinit = 50)
stn.unit('Reactor_1', 'Reaction_2', Bmin = 0, Bmax =  80, tm = 21)
stn.unit('Reactor_1', 'Reaction_3', Bmin = 0, Bmax =  80, tm = 21)
stn.unit('Reactor_2', 'Reaction_1', Bmin = 0, Bmax =  50, tm = 24, rmax = 160, rinit = 120)
stn.unit('Reactor_2', 'Reaction_2', Bmin = 0, Bmax =  50, tm = 24)
stn.unit('Reactor_2', 'Reaction_3', Bmin = 0, Bmax =  50, tm = 24)
stn.unit('Still',     'Separation', Bmin = 0, Bmax = 200, tm = 15, rmax = 100, rinit = 40)

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

stn.demand('Product_1', 168, 150)
stn.demand('Product_2', 168, 200) 

H = 168
dH = 3
stn.build(range(0,H+1,dH))
stn.solve('glpk')
stn.gantt()
stn.trace()
