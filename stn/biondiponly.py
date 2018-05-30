#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import dill
import time
import scipy.stats as sct
import numpy as np
import pandas as pd
sys.path.append('../STN/modules')
from stn import stnModel, stnModelRobust # noqa
from blocks import blockPlanning  # noqa
import deg  # noqa

rdir = "/home/jw3617/STN/results"
pref = "low_"
TPfile = "../data/biondi3TP.pkl"
# TPfile = "../data/TP.pkl"
Q = np.arange(0.02, 0.51, 0.02)
# Q = [0.35]

# Load results files
cols = ["ID", "alpha", "epsilon", "Preactor1",
        "Preactor2", "Pheater", "Pstill", "CostStorage",
        "CostMaintenance", "Cost", "Cost0", "Dslack", "timeTotal",
        "demand1", "demand2", "inf", "gapmax", "gapmean", "gapmin"]
df = pd.DataFrame(columns=cols)
rid = -1
try:
    df2 = pd.read_pickle(rdir+"/results.pkl")
    rid = max(df2["ID"])
    df = df.append(df2)
except IOError:
    pass

demand_1 = [150, 88, 125, 67, 166, 203, 90, 224,
            174, 126, 66, 119, 234, 64,
            103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
demand_2 = [200, 150, 197, 296, 191, 193, 214,
            294, 247, 313, 226, 121, 197,
            242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]

np.random.seed(12)
demand_1 = np.random.uniform(size=24)*(250-180) + 180
demand_2 = np.random.uniform(size=24)*(350-270) + 270
np.random.seed(42)
demand_1 = np.random.uniform(size=24)*(120-50) + 50
demand_2 = np.random.uniform(size=24)*(180-100) + 100

for n, q in enumerate(Q):
    # create instance
    rid += 1
    t = time.time()
    with open("../data/biondiM.dat", "rb") as dill_file:
        stn = dill.load(dill_file)

    Ts = 168
    dTs = 3
    Tp = 168*24
    dTp = 168
    TIMEs = range(0, Ts, dTs)
    TIMEp = range(0, Tp, dTp)

    # model = stnModel(stn)
    # model = stnModelRobust(stn)
    model = blockPlanning(stn, [0, Tp, dTp])

    for i in range(0, len(TIMEp)):
        model.demand('Product_1', TIMEp[i], demand_1[i])
        model.demand('Product_2', TIMEp[i], demand_2[i])
    model.build(objective="terminal", decisionrule="continuous", alpha=q,
                rdir=rdir, prefix=pref, rid=rid)

    eps = 1 - sct.norm.ppf(q=q, loc=1, scale=0.27)
    solverparams = {"timelimit": 120,
                    "mipgap": 0.02}
    # model.solve(  # [Ts, dTs, Tp, dTp],
    #             solver="cplex",
    #             prefix="biondiD_"+str(rid),
    #             rdir="/home/jw3617/STN/results",
    #             tindexed=False,
    #             trace=True,
    #             save=True,
    #             solverparams=solverparams)
    model.loadres("/home/jw3617/STN/results_biondi_mc_lowD/biondiD_"
                  + str(rid)+"STN.pyomo")
    pdfloc = model.calc_p_fail(TP=TPfile, periods=12, Nmc=100)
    preactor1 = max(pdfloc["Reactor_1"])
    preactor2 = max(pdfloc["Reactor_2"])
    pheater = max(pdfloc["Heater"])
    pstill = max(pdfloc["Still"])
    cost_storage = 0
    cost_maint = 0
    cost = 0
    dslack = 0
    for t in TIMEp:
        cost_storage += model.b.CostStorage[t]()
        cost_maint += model.b.CostMaintenance[t]()
    cost += (cost_storage + cost_maint
             + model.b.CostMaintenanceFinal())
    # cost0 = model.m_list[0].Obj()
    cost0 = model.model.Obj()
    ttot = time.time() - t
    demand1 = sum(demand_1)
    demand2 = sum(demand_2)
    inf = False
    gapmean = 0  # model.solver._gap/cost0
    gapmax = 0  # model.solver._gap/cost0
    gapmin = 0  # model.solver._gap/cost0
    line = pd.DataFrame([[rid, q, eps, preactor1, preactor2,
                          pheater, pstill, cost_storage,
                         cost_maint, cost, cost0, dslack,
                         ttot, demand1, demand2, inf,
                         gapmax, gapmean, gapmin]], columns=cols)
    df = df.append(line, ignore_index=True)
    print(df)
df.to_pickle(rdir+"/results.pkl")
df.to_csv(rdir+"/results.csv")
