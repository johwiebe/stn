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
from stn import stnModelRobust # noqa
import deg  # noqa

rdir = "/home/jw3617/STN/results"
Q = np.arange(0.05, 0.51, 0.05)
Q = [0.5]
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
pcols = ["id", "Reactor_1", "Reactor_2", "Heater", "Still"]
pdf = pd.DataFrame(columns=pcols)
try:
    pdf2 = pd.read_pickle(rdir+"/pfail.pkl")
    pdf.append(pdf2)
except IOError:
    pass
Nsim = 1
Pheater = np.ones((len(Q), Nsim))
Preactor = np.ones((len(Q), Nsim))
for n, q in enumerate(Q):
    # create instance
    rid += 1
    t = time.time()
    model = stnModelRobust()
    with open("biondiR.dat", "rb") as dill_file:
        stn = dill.load(dill_file)

    model.stn = stn

    demand_1 = [150, 88, 125, 67, 166, 203, 90, 224,
                174, 126, 66, 119, 234, 64,
                103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
    demand_2 = [200, 150, 197, 296, 191, 193, 214,
                294, 247, 313, 226, 121, 197,
                242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]

    Ts = 168
    dTs = 3
    Tp = 168*24
    dTp = 168
    TIMEs = range(0, Ts, dTs)
    TIMEp = range(0, Tp, dTp)
    for i in range(0, len(TIMEp)):
        model.demand('Product_1', TIMEp[i], demand_1[i])
        model.demand('Product_2', TIMEp[i], demand_2[i])

    eps = 1 - sct.norm.ppf(q=q, loc=1, scale=0.27)
    print(eps)
    model.uncertainty(q)
    solverparams = {"timelimit": 600,
                    "mipgap": 0.02}
    import ipdb; ipdb.set_trace()  # noqa
    model.solve([Ts, dTs, Tp, dTp],
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                # periods=12,
                prefix="biondiD"+str(rid),
                rdir="/home/jw3617/STN/results",
                tindexed=False,
                save=True,
                solverparams=solverparams)
    import ipdb; ipdb.set_trace()  # noqa
    # preactor1 = deg.simulate_deg(100000, model, "Reactor_1", Sinit=100, dt=3)
    # preactor2 = deg.simulate_deg(100000, model, "Reactor_2", Sinit=40, dt=3)
    # pheater = deg.simulate_deg(100000, model, "Heater", Sinit=50, dt=3)
    # pstill = deg.simulate_deg(100000, model, "Still", Sinit=60, dt=3)
    pr1 = deg.calc_p_fail(model, "Reactor_1", q, "../data/TP.pkl", pb=True,
                          periods=12)
    pr2 = deg.calc_p_fail(model, "Reactor_2", q, "../data/TP.pkl", pb=True,
                          periods=12)
    ph = deg.calc_p_fail(model, "Heater", q, "../data/TP.pkl", pb=True,
                         periods=12)
    ps = deg.calc_p_fail(model, "Still", q, "../data/TP.pkl", pb=True,
                         periods=12)
    pdfloc = pd.DataFrame(np.transpose(np.array([pr1, pr2, ph, ps])),
                          columns=["Reactor_1", "Reactor_2",
                                   "Heater", "Still"])
    pdfloc["id"] = rid
    pdf = pdf.append(pdfloc)
    preactor1 = max(pr1)
    preactor2 = max(pr2)
    pheater = max(ph)
    pstill = max(ps)
    print("eps: " + str(eps) + ", P: " + str(np.mean(pr1)))
    for j in model.stn.units:
        profile = model.get_unit_profile(j, full=False)
        profile["id"] = rid
        dfp = dfp.append(profile)
    dfp = dfp.append(profile)
    print("MCS Reactor")
    cost_storage = 0
    cost_maint = 0
    cost = 0
    dslack = 0
    for period, m in enumerate(model.m_list):
        cost_storage += m.sb.CostStorage()
        cost_maint += m.sb.CostMaintenance()
        dslack += m.TotSlack()
    cost += cost_storage + cost_maint + m.pb.CostMaintenanceFinal()
    cost0 = model.m_list[0].Obj()
    ttot = time.time() - t
    demand1 = sum(demand_1)
    demand2 = sum(demand_2)
    inf = len(model.m_list) < 12
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
pdf.to_pickle(rdir+"/pfail.pkl")
pdf.to_csv(rdir+"/pfail.csv")
