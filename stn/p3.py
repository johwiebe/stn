#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import dill
sys.path.append('../STN/modules')
from stn import stnModel # noqa
import deg  # noqa

rdir = "/home/jw3617/STN/results"
demand_1 = [250, 188, 325, 267, 166, 303, 290, 224,
            174, 326, 266, 219, 334, 164,
            203, 277, 332, 286, 174, 239, 324, 194, 291, 228]
demand_2 = [250, 188, 325, 267, 166, 303, 290, 224,
            174, 326, 266, 219, 334, 164,
            203, 277, 332, 286, 174, 239, 324, 194, 291, 228]
# create instance
with open("../data/p3D.dat", "rb") as dill_file:
    stn = dill.load(dill_file)
model = stnModel(stn)

Ts = 168
dTs = 3
Tp = 168*24
dTp = 168
TIMEs = range(0, Ts, dTs)
TIMEp = range(0, Tp, dTp)
for i in range(0, len(TIMEp)):
    model.demand('S12', TIMEp[i], demand_1[i])
    model.demand('S13', TIMEp[i], demand_2[i])

solverparams = {"timelimit": 360}
model.solve([Ts, dTs, Tp, dTp],
            solver="cplex",
            objective="terminal",
            decisionrule="continuous",
            # periods=12,
            prefix="p2D",
            rdir="/home/jw3617/STN/results",
            tindexed=False,
            trace=True,
            save=True,
            solverparams=solverparams)
