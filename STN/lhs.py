#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve model by Biondi et al. with scheduling block only
"""

import sys
import dill
import yaml
import pyDOE
import numpy as np
sys.path.append('../STN/modules')
from blocks import blockScheduling # noqa

with open(sys.argv[1], "r") as f:
    y = yaml.load(f)

with open(y["stn"], "rb") as dill_file:
    stn = dill.load(dill_file)

# define demand
N = y["N"]
mlhs = pyDOE.lhs(len(stn.products), samples=y["N"], criterion="maximin")
dem = {}
for p in stn.products:
    dem[p] = (np.array([mlhs[i][0] for i in range(0, N)])
              * (y["max"][p] - y["min"][p])
              + y["min"][p])

for i in range(0, N):
    for j in stn.units:
        stn.Rinit[j] = 0
    dem_i = {}
    for p in stn.products:
        dem_i[p] = dem[p][i]
    model = blockScheduling(stn, [0, y["Ts"], y["dTs"]],
                            dem_i)
    model.build(rdir=y["rdir"])
    model.solve(
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                tindexed=False,
                save=True,
                trace=True,
                solverparams=y["solverparams"])

    if not model.inf:
        dfp = model.get_unit_profile()
    df = model.eval()

    with open(y["stn"], "rb") as dill_file:
        stn = dill.load(dill_file)
