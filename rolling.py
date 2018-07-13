#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solve STN model using rolling horizon."""

import sys
import yaml
import dill

from stn import stnModel, stnModelRobust # noqa


with open(sys.argv[1], "r") as f:
    y = yaml.load(f)

# time horizons
TIMEp = range(0, y["Tp"], y["dTp"])

for n, q in enumerate(y["alphas"]):
    with open(y["stn"], "rb") as dill_file:
        stn = dill.load(dill_file)

    if y["robust"]:
        model = stnModelRobust(stn)
    else:
        model = stnModel(stn)
    for i, t in enumerate(TIMEp):
        for p in stn.products:
            model.demand(p, t, y[p][i])

# build and solve model
    model.solve([y["Ts"], y["dTs"], y["Tp"], y["dTp"]],
                solver="cplex",
                objective="terminal",
                periods=y["periods"]["rolling"],
                prefix=y["prfx"],
                rdir=y["rdir"],
                save=True,
                alpha=q,
                trace=True,
                solverparams=y["solverparams"],
                tindexed=False)
    model.eval(periods=y["periods"]["eval"], TP=y["TP"])