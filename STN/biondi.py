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
TPfile = "../data/biondi2TP.pkl"
# Q = np.arange(0.05, 0.51, 0.05)
Q = np.arange(0.23, 0.51, 0.03)
# Q = [0.5, 0.5, 0.5]

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
dcols = ["id", "period", "time", "unit", "task", "mode", "P1", "P2"]
dfp = pd.DataFrame(columns=dcols)
try:
    dfp2 = pd.read_pickle(rdir+"/profile.pkl")
except IOError:
    dfp2 = dfp
pcols = ["id", "eps", "alpha", "Reactor_1", "Reactor_2", "Heater", "Still"]
pdf = pd.DataFrame(columns=pcols)
try:
    pdf2 = pd.read_pickle(rdir+"/pfail.pkl")
    pdf.append(pdf2)
except IOError:
    pass

# demand_1 = [150, 88, 125, 67, 166, 203, 90, 224,
#             174, 126, 66, 119, 234, 64,
#             103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
# demand_2 = [200, 150, 197, 296, 191, 193, 214,
#             294, 247, 313, 226, 121, 197,
#             242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]
np.random.seed(12)
demand_1 = np.random.uniform(size=24)*(250-180) + 180
demand_2 = np.random.uniform(size=24)*(350-270) + 270
# np.random.seed(42)
# demand_1 = np.random.uniform(size=24)*(120-50) + 50
# demand_2 = np.random.uniform(size=24)*(180-100) + 100

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

    model = stnModel(stn)
    # model = stnModelRobust(stn)

    for i in range(0, len(TIMEp)):
        model.demand('Product_1', TIMEp[i], demand_1[i])
        model.demand('Product_2', TIMEp[i], demand_2[i])

    eps = 1 - sct.norm.ppf(q=q, loc=1, scale=0.27)
    solverparams = {"timelimit": 120,
                    "mipgap": 0.02}
    model.solve([Ts, dTs, Tp, dTp],
                solver="cplex",
                # objective=ob[n],
                objective="terminal",
                decisionrule="continuous",
                periods=12,
                prefix="biondiD_"+str(rid),
                rdir="/home/jw3617/STN/results",
                tindexed=False,
                trace=True,
                save=True,
                alpha=q,
                solverparams=solverparams)
    # model.build([Ts, dTs, Tp, dTp], period=11, alpha=q)
    # model.loadres(prefix="/home/jw3617/STN/old/results_biondi_25Q_9R_bigD/biondiD"
    #               + str(rid)+"_",
    #               periods=12)
    pdfloc = model.calc_p_fail(TP=TPfile, periods=12)
    pdfloc["id"] = rid
    pdfloc["eps"] = eps
    pdfloc["alpha"] = q
    pdf = pdf.append(pdfloc)
    preactor1 = np.mean(pdfloc["Reactor_1"])
    preactor2 = np.mean(pdfloc["Reactor_2"])
    pheater = np.mean(pdfloc["Heater"])
    pstill = np.mean(pdfloc["Still"])
    for j in model.stn.units:
        profile = model.get_unit_profile(j, full=False)
        profile["id"] = rid
        dfp = dfp.append(profile)
    cost_storage = 0
    cost_maint = 0
    cost = 0
    dslack = 0
    for period, m in enumerate(model.m_list):
        cost_storage += m.sb.CostStorage()
        cost_maint += m.sb.CostMaintenance()
        dslack += m.TotSlack()
    cost += (cost_storage + cost_maint
             + model.sb.get_cost_maintenance_terminal())
    cost0 = model.m_list[0].Obj()
    ttot = time.time() - t
    demand1 = sum(demand_1)
    demand2 = sum(demand_2)
    inf = False
    gapmax, gapmean, gapmin = model.get_gap()
    line = pd.DataFrame([[rid, q, eps, preactor1, preactor2,
                          pheater, pstill, cost_storage,
                         cost_maint, cost, cost0, dslack,
                         ttot, demand1, demand2, inf,
                         gapmax, gapmean, gapmin]], columns=cols)
    df = df.append(line, ignore_index=True)
    print(df)
df.to_pickle(rdir+"/results.pkl")
df.to_csv(rdir+"/results.csv")
dfp = dfp.append(dfp2)
dfp.to_pickle(rdir+"/profile.pkl")
dfp.to_csv(rdir+"/profile.csv")
# pdf.to_pickle(rdir+"/pfail.pkl")
# pdf.to_csv(rdir+"/pfail.csv")
