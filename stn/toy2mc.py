#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import dill
import yaml
sys.path.append('../STN/modules')
from blocks import blockPlanning  # noqa
import deg  # noqa

with open(sys.argv[1], "r") as f:
    y = yaml.load(f)
with open(y["stn"], "rb") as dill_file:
    stn = dill.load(dill_file)
Ts = y["Ts"]
dTs = y["dTs"]
Tp = y["Tp"]
dTp = y["dTp"]
TIMEp = range(0, Tp, dTp)
for n, q in enumerate(y["alphas"]):
    # create instance

    model = blockPlanning(stn, [0, Tp, dTp])
    for i in range(0, len(TIMEp)):
        for p in stn.products:
            model.demand(p, TIMEp[i], y[p][i])
    model.build(objective="terminal", decisionrule="continuous", alpha=q,
                rdir=y["rdir"], prefix=y["prfx"])

    model.solve(  # [Ts, dTs, Tp, dTp],
                solver="cplex",
                tindexed=False,
                trace=True,
                save=True,
                solverparams=y["solverparams"])
    # model.loadres("/home/jw3617/STN/results_biondi_mc_lowD/biondiD_"
    #               + str(rid)+"STN.pyomo")
    # pdfloc = model.calc_p_fail(TP=TPfile, periods=12, Nmc=100)
    df = model.eval(TP=y["TP"], periods=12)
