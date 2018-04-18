#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import dill
sys.path.append('../STN/modules')
from blocks2 import blockPlanningRobust # noqa

# create instance

with open("biondiRstruct.dat", "rb") as dill_file:
    stn = dill.load(dill_file)


demand_1 = [150, 88, 125, 67, 166, 203, 90, 224, 174, 126, 66, 119, 234, 64,
            103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
demand_2 = [200, 150, 197, 296, 191, 193, 214, 294, 247, 313, 226, 121, 197,
            242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]
Ts = 168
dTs = 3
Tp = 168*24
dTp = 168
TIMEs = range(0, Ts, dTs)
TIMEp = range(0, Tp, dTp)
# demand_1 = [100 for i in range(0, Tp, dTp)]
# demand_1 = [100 for i in range(0, Tp, dTp)]

model = blockPlanningRobust(stn, [0, Tp, dTp],
                            {"Product_1": 150, "Product_2": 200})
for n, t in enumerate(TIMEp):
    model.demand('Product_1', t, demand_1[n])
    model.demand('Product_2', t, demand_2[n])

model.build()

solverparams = {"timelimit": 60, "mipgap": 0.02,
                "mip_strategy_heuristicfreq": 10}
# self.solver.options["mip_strategy_heuristicfreq"] = 10
model.solve(
            solver="cplex",
            objective="terminal",
            decisionrule="continuous",
            prefix="biondiR",
            rdir="/home/jw3617/STN/results",
            tindexed=False,
            save=True,
            trace=True,
            solverparams=solverparams)

import ipdb; ipdb.set_trace()  # noqa
