#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve model by Biondi et al. with scheduling block only repeatedly.
    config: .yaml configuration file
"""

import sys
import dill
import yaml
import pyDOE
import numpy as np
from stn.blocks import blockScheduling # noqa

# Load configuration file
with open(sys.argv[1], "r") as f:
    y = yaml.load(f)
# Load stn structure
with open(y["stn"], "rb") as dill_file:
    stnstruct = dill.load(dill_file)

# Define demand using LHS
N = y["N"]
mlhs = pyDOE.lhs(len(stnstruct.products), samples=y["N"], criterion="maximin")
dem = {}
for p_ind, p in enumerate(stnstruct.products):
    dem[p] = (np.array([mlhs[i][p_ind] for i in range(0, N)])
              * (y["max"][p] - y["min"][p])
              + y["min"][p])

# Solve for each demand tupel
for i in range(0, N):
    for j in stnstruct.units:
        stnstruct.Rinit[j] = 0
    dem_i = {}
    for p in stnstruct.products:
        dem_i[p] = dem[p][i]
    # Initialize scheduling only model
    model = blockScheduling(stnstruct, [0, y["Ts"], y["dTs"]],
                            dem_i)
    model.build(rdir=y["rdir"])
    # Solve
    model.solve(
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                tindexed=False,
                save=True,
                trace=True,
                solverparams=y["solverparams"])
    # Evaluate
    if not model.inf:
        dfp = model.get_unit_profile()  # save task profile/schedule
    df = model.eval()

    with open(y["stn"], "rb") as dill_file:
        stnstruct = dill.load(dill_file)
