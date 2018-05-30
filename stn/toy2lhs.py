#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve model by Biondi et al. with scheduling block only
"""

import sys
import dill
import pyDOE
import numpy as np
sys.path.append('../STN/modules')
from blocks import blockScheduling # noqa


# define demand
N = 200
mlhs = pyDOE.lhs(2, samples=N, criterion="maximin")
demand_1 = np.array([mlhs[i][0] for i in range(0, N)])*(250)
demand_2 = np.array([mlhs[i][1] for i in range(0, N)])*(250)

# define time step and horizon
Ts = 30
dTs = 1

# folders and files
rdir = "/home/jw3617/STN/results_toy2/LHS"
stnfile = "../data/toy2.dat"

for i in range(0, N):
    with open(stnfile, "rb") as dill_file:
        stn = dill.load(dill_file)
    for j in stn.units:
        stn.Rinit[j] = 0
    model = blockScheduling(stn, [0, Ts, dTs],
                            {"P1": demand_1[i],
                             "P2": demand_2[i]})
    model.build(rdir=rdir)
    solverparams = {
                    "timelimit": 60,
                   }
    model.solve(
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                tindexed=False,
                save=True,
                trace=True,
                solverparams=solverparams)

    if not model.inf:
        dfp = model.get_unit_profile()
    df = model.eval()
