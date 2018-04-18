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
from stn import stnModel # noqa
import deg  # noqa

rdir = "/home/jw3617/STN/results"
Q = np.arange(0.02, 0.51, 0.02)
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
pcols = ["id", "eps", "alpha", "Reactor_1", "Reactor_2", "Heater", "Still"]
pdf = pd.DataFrame(columns=pcols)
try:
    pdf2 = pd.read_pickle(rdir+"/pfail.pkl")
    pdf.append(pdf2)
except IOError:
    pass
Nsim = 1
Pheater = np.ones((len(Q), Nsim))
Preactor = np.ones((len(Q), Nsim))
demand_1 = [150, 88, 125, 67, 166, 203, 90, 224,
            174, 126, 66, 119, 234, 64,
            103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
demand_2 = [200, 150, 197, 296, 191, 193, 214,
            294, 247, 313, 226, 121, 197,
            242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]
# demand_1 = [50, 88, 75, 67, 66, 53, 90, 84,
#             74, 100, 66, 59, 64, 64,
#             93, 77, 52, 86, 74, 89, 54, 94, 91, 78]
# demand_2 = [100, 150, 127, 146, 111, 133, 114,
#             124, 147, 113, 126, 121, 107,
#             142, 120, 142, 150, 120, 135, 108, 122, 122, 124, 137]

for n, q in enumerate(Q):
    # create instance
    rid += 1
    t = time.time()
    model = stnModel()
    with open("biondistruct.dat", "rb") as dill_file:
        stn = dill.load(dill_file)
    eps = 1 - sct.norm.ppf(q=q, loc=1, scale=0.27)
    stn.ijkdata('Heating', 'Heater', 'Slow', 9, 1*(1+eps))
    stn.ijkdata('Heating', 'Heater', 'Normal', 6, 2*(1+eps))
    stn.ijkdata('Heating', 'Heater', 'Fast', 3, 3*(1+eps))
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Slow', 27, 4*(1+eps))
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Normal', 15, 5*(1+eps))
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Fast', 9, 8*(1+eps))
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Slow', 30, 4*(1+eps))
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Normal', 18, 5*(1+eps))
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Fast', 12, 10*(1+eps))
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Slow', 36, 1*(1+eps))
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Normal', 21, 3*(1+eps))
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Fast', 15, 5*(1+eps))
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Slow', 33, 2*(1+eps))
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Normal', 18, 4*(1+eps))
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Fast', 12, 4*(1+eps))
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Slow', 30, 3*(1+eps))
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Normal', 18, 7*(1+eps))
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Fast', 6, 8*(1+eps))
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Slow', 24, 2*(1+eps))
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Normal', 21, 5*(1+eps))
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Fast', 12, 9*(1+eps))
    stn.ijkdata('Separation', 'Still', 'Slow', 15, 2*(1+eps))
    stn.ijkdata('Separation', 'Still', 'Normal', 9, 5*(1+eps))
    stn.ijkdata('Separation', 'Still', 'Fast', 6, 6*(1+eps))
    model.stn = stn

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
    model.uncertainty(eps)
    solverparams = {"timelimit": 600,
                    }  # "mipgap": 0.02}
    model.solve([Ts, dTs, Tp, dTp],
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                # periods=12,
                prefix="biondiD"+str(rid),
                rdir="/home/jw3617/STN/results",
                tindexed=False,
                trace=True,
                save=True,
                solverparams=solverparams)
    with open("biondiRstruct.dat", "rb") as dill_file:
        model.stn = dill.load(dill_file)
    # preactor1 = deg.simulate_deg(100000, model, "Reactor_1", Sinit=100, dt=1)
    # preactor2 = deg.simulate_deg(100000, model, "Reactor_2", Sinit=40, dt=1)
    # pheater = deg.simulate_deg(100000, model, "Heater", Sinit=50, dt=1)
    # pstill = deg.simulate_deg(100000, model, "Still", Sinit=60, dt=1)
    pr1 = deg.simulate_deg_pb(1000, 100, model, "Reactor_1", eps, pb=True,
                              periods=12)
    pr2 = deg.simulate_deg_pb(1000, 100, model, "Reactor_2", eps, pb=True,
                              periods=12)
    ph = deg.simulate_deg_pb(1000, 100, model, "Heater", eps, pb=True,
                             periods=12)
    ps = deg.simulate_deg_pb(1000, 100, model, "Still", eps, pb=True,
                             periods=12)
    pdfloc = pd.DataFrame(np.transpose(np.array([pr1, pr2, ph, ps])),
                          columns=["Reactor_1", "Reactor_2",
                                   "Heater", "Still"])
    pdfloc["id"] = rid
    pdfloc["eps"] = eps
    pdfloc["alpha"] = q
    pdf = pdf.append(pdfloc)
    preactor1 = max(pr1)
    preactor2 = max(pr2)
    pheater = max(ph)
    pstill = max(ps)
    print("eps: " + str(eps) + ", P: " + str(preactor1))
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
