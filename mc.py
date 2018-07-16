#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve planning only model and estimate p^f_j using Markov-chain or Frequency
approach.
    config: .yaml config file
"""

import sys
import dill
import yaml
from stn import blockPlanning  # noqa
import stn.deg as deg # noqa

# Load config file
with open(sys.argv[1], "r") as f:
    y = yaml.load(f)
# Load STN structure
with open(y["stn"], "rb") as dill_file:
    stn = dill.load(dill_file)
Ts = y["Ts"]
dTs = y["dTs"]
Tp = y["Tp"]
dTp = y["dTp"]
TIMEp = range(0, Tp, dTp)
# Solve model for each alpha
for n, q in enumerate(y["alphas"]):
    # Create instance
    model = blockPlanning(stn, [0, Tp, dTp])
    for i in range(0, len(TIMEp)):
        for p in stn.products:
            model.demand(p, TIMEp[i], y[p][i])
    model.build(objective="terminal", decisionrule="continuous", alpha=q,
                rdir=y["rdir"], prefix=y["prfx"])
    # Solve instance
    model.solve(
                solver="cplex",
                tindexed=False,
                trace=True,
                save=True,
                solverparams=y["solverparams"])
    # Evaluate
    df = model.eval(TP=y["TP"], periods=y["periods"], dTs=y["dTs"])
