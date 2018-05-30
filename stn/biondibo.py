#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import pandas as pd
import time
import scipy.stats as sct
from skopt import gp_minimize
sys.path.append('../STN/modules')
from stn import stnModelRobust # noqa
import deg  # noqa


def target(x):
    q = x[0]
    rdir = "/home/jw3617/STN/results"
    cols = ["ID", "alpha", "epsilon", "Preactor", "CostStorage",
            "CostMaintenance", "Cost", "Cost0", "Dslack", "timeTotal"]
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
    rid += 1
    t = time.time()
    # create instance
    model = stnModelRobust()
    stn = model.stn

    # states
    stn.state('FeedA',     init=2000000)
    stn.state('FeedB',     init=2000000)
    stn.state('FeedC',     init=2000000)
    stn.state('HotA',      price=-1, capacity=100, scost=1)
    stn.state('IntAB',     price=-1, capacity=200, scost=1)
    stn.state('IntBC',     price=-1, capacity=150, scost=1)
    stn.state('ImpureE',   price=-1, capacity=100, scost=1)
    stn.state('Product_1', price=10, scost=5)
    stn.state('Product_2', price=10, scost=5)
    # state to task arcs
    stn.stArc('FeedA',   'Heating')
    stn.stArc('FeedB',   'Reaction_1', rho=0.5)
    stn.stArc('FeedC',   'Reaction_1', rho=0.5)
    stn.stArc('FeedC',   'Reaction_3', rho=0.2)
    stn.stArc('HotA',    'Reaction_2', rho=0.4)
    stn.stArc('IntAB',   'Reaction_3', rho=0.8)
    stn.stArc('IntBC',   'Reaction_2', rho=0.6)
    stn.stArc('ImpureE', 'Separation')
    # task to state arcs
    stn.tsArc('Heating',    'HotA',      rho=1.0)
    stn.tsArc('Reaction_2', 'IntAB',     rho=0.6)
    stn.tsArc('Reaction_2', 'Product_1', rho=0.4)
    stn.tsArc('Reaction_1', 'IntBC')
    stn.tsArc('Reaction_3', 'ImpureE')
    stn.tsArc('Separation', 'IntAB',     rho=0.1)
    stn.tsArc('Separation', 'Product_2', rho=0.9)
    # unit-task data
    stn.unit('Heater',    'Heating',    Bmin=40, Bmax=100, tm=15, rmax=80,
             rinit=50, a=600, b=300)
    stn.unit('Reactor_1', 'Reaction_1', Bmin=32, Bmax=80, tm=21, rmax=150,
             rinit=100, a=1500, b=600)
    stn.unit('Reactor_1', 'Reaction_2', Bmin=32, Bmax=80, tm=21)
    stn.unit('Reactor_1', 'Reaction_3', Bmin=32, Bmax=80, tm=21)
    stn.unit('Reactor_2', 'Reaction_1', Bmin=20, Bmax=50, tm=24, rmax=160,
             rinit=40, a=2500, b=500)
    stn.unit('Reactor_2', 'Reaction_2', Bmin=20, Bmax=50, tm=24)
    stn.unit('Reactor_2', 'Reaction_3', Bmin=20, Bmax=50, tm=24)
    stn.unit('Still',     'Separation', Bmin=80, Bmax=200, tm=15, rmax=100,
             rinit=60, a=2700, b=1500)
    stn.opmode('Slow')
    stn.opmode('Normal')
    stn.opmode('Fast')
    stn.ijkdata('Heating', 'Heater', 'Slow', 9, 1)
    stn.ijkdata('Heating', 'Heater', 'Normal', 6, 2)
    stn.ijkdata('Heating', 'Heater', 'Fast', 3, 3)
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Slow', 27, 4)
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Normal', 15, 5)
    stn.ijkdata('Reaction_1', 'Reactor_1', 'Fast', 9, 8)
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Slow', 30, 4)
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Normal', 18, 5)
    stn.ijkdata('Reaction_1', 'Reactor_2', 'Fast', 12, 10)
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Slow', 36, 1)
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Normal', 21, 3)
    stn.ijkdata('Reaction_2', 'Reactor_1', 'Fast', 15, 5)
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Slow', 33, 2)
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Normal', 18, 4)
    stn.ijkdata('Reaction_2', 'Reactor_2', 'Fast', 12, 4)
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Slow', 30, 3)
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Normal', 18, 7)
    stn.ijkdata('Reaction_3', 'Reactor_1', 'Fast', 6, 8)
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Slow', 24, 2)
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Normal', 21, 5)
    stn.ijkdata('Reaction_3', 'Reactor_2', 'Fast', 12, 9)
    stn.ijkdata('Separation', 'Still', 'Slow', 15, 2)
    stn.ijkdata('Separation', 'Still', 'Normal', 9, 5)
    stn.ijkdata('Separation', 'Still', 'Fast', 6, 6)

    demand_1 = [150, 88, 125, 67, 166, 203, 90, 224, 174, 126, 66, 119,
                234, 64, 103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
    demand_2 = [200, 150, 197, 296, 191, 193, 214, 294, 247, 313, 226,
                121, 197, 242, 220, 342, 355, 320, 335, 298, 252, 222,
                324, 337]

    Ts = 168
    dTs = 3
    Tp = 168*24
    dTp = 168
    TIMEp = range(0, Tp, dTp)

    for i in range(0, len(TIMEp)):
        model.demand('Product_1', TIMEp[i], demand_1[i])
        model.demand('Product_2', TIMEp[i], demand_2[i])

    eps = 1 - sct.norm.ppf(q=q, loc=1, scale=0.27)
    model.uncertainty(eps)
    solverparams = {"timelimit": 120,
                    "mipgap": 0.1}
    model.solve([Ts, dTs, Tp, dTp],
                solver="cplex",
                objective="terminal",
                decisionrule="continuous",
                periods=1,
                prefix="biondiR"+str(q),
                rdir="/home/jw3617/STN/results",
                tindexed=False,
                solverparams=solverparams)

    preactor = deg.simulate_deg(20000, model, "Reactor_1", Sinit=100, dt=1)
    profile = model.get_unit_profile("Reactor_1")
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
    line = [rid, q, eps, preactor, cost_storage,
            cost_maint, cost, cost0, dslack,
            ttot]
    df.loc[rid] = line
    print(df)
    df.to_pickle(rdir+"/results.pkl")
    df.to_csv(rdir+"/results.csv")
    dfp = dfp.append(dfp2)
    dfp.to_pickle(rdir+"/profile.pkl")
    dfp.to_csv(rdir+"/profile.csv")

    return preactor/100*70000 + (1-preactor/100)*cost


if __name__ == "__main__":

    bo = gp_minimize(target, [(0.05, 0.5)], acq_func="EI", n_calls=5,
                     n_random_starts=5, noise=0.2)
