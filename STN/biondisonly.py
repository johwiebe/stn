#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve model by Biondi et al. with scheduling block only
"""

import sys
import time
import dill
import pyDOE
import pandas as pd
import numpy as np
sys.path.append('../STN/modules')
from blocks import blockScheduling # noqa

# create instance

with open("../data/biondiR.dat", "rb") as dill_file:
    stn = dill.load(dill_file)

N = 200
mlhs = pyDOE.lhs(2, samples=N, criterion="maximin")
demand_1 = 50 + np.array([mlhs[i][0] for i in range(0, N)])*(250-50)
demand_2 = 100 + np.array([mlhs[i][1] for i in range(0, N)])*(350-100)

for j in stn.units:
    stn.Rinit[j] = 0
# stn.Rinit["Reactor_1"] = stn.Rmax["Reactor_1"]

Ts = 168
dTs = 3

rdir = "/home/jw3617/STN/results"
cols = ["ID", "CostStorage",
        "CostMaintenance", "CostMaintenanceFinal", "Obj",
        "Infeasible", "timeTotal",
        "Demand1", "Demand2", "Gap"]
df = pd.DataFrame(columns=cols)
rid = -1
try:
    df2 = pd.read_pickle(rdir+"/results.pkl")
    rid = max(df2["ID"])
    df = df.append(df2)
except IOError:
    pass
dcols = ["id", "unit", "time", "gap", "unit", "task", "mode", "P1", "P2"]
dfp = pd.DataFrame(columns=dcols)
dfp = pd.DataFrame()
try:
    dfp2 = pd.read_pickle(rdir+"/profile.pkl")
except IOError:
    dfp2 = dfp

N = 1
for i in range(0, N):
    rid += 1
    t = time.time()
    model = blockScheduling(stn, [0, Ts, dTs],
                            {"Product_1": demand_1[i],
                             "Product_2": demand_2[i]})
    model.build()
    # model.uncertainty(0.1)
    solverparams = {
                    "timelimit": 30,
                    # "dettimelimit": 72000
                    "mipgap": 0.02,
                    # "mip_strategy_heuristicfreq": 30
                   }
    model.solve(
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                prefix="biondiSD_"+str(rid),
                rdir="/home/jw3617/STN/results",
                tindexed=False,
                save=True,
                trace=True,
                solverparams=solverparams)

    ttot = time.time() - t
    inf = model.inf
    if not inf:
        gap = model.solver._gap/model.b.Obj()*100
        for j in model.stn.units:
            profile = model.get_unit_profile(j, full=False)
            profile["id"] = rid
            profile["gap"] = gap
            dfp = dfp.append(profile)
        cost_storage = model.b.CostStorage()
        cost_maint = model.b.CostMaintenance()
        cost_maint_final = model.b.CostMaintenanceFinal()
        obj = model.b.Obj()
        line = pd.DataFrame([[rid, cost_storage,
                             cost_maint, cost_maint_final, obj, inf,
                             ttot, demand_1[i], demand_2[i], gap]],
                            columns=cols)
    else:
        line = pd.DataFrame([[rid, 0, 0, 0, "Inf", inf, ttot, demand_1[i],
                              demand_2[i], "NA"]], columns=cols)
    df = df.append(line, ignore_index=True)
    print(df)
df.to_pickle(rdir+"/results.pkl")
df.to_csv(rdir+"/results.csv")
dfp = dfp.append(dfp2)
dfp.to_pickle(rdir+"/profile.pkl")
dfp.to_csv(rdir+"/profile.csv")
