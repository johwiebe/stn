#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Toy example of STN with degradation.

Units:
    Heater
    Reactor

Tasks:
    Heating
    Reaction_1
    Reaction_2

Operating modes:
    Slow
    Normal

"""

import sys
import time
import scipy.stats as sct
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
sys.path.append('../STN/modules')

from stn import stnModelRobust # noqa
import deg # noqa


# create instance
def target(x):
    q = x[0]
    rdir = "/home/jw3617/STN/results"
    cols = ["ID", "alpha", "epsilon", "Pheater", "Preactor", "CostStorage",
            "CostMaintenance", "Cost", "Cost0", "Dslack", "timeTotal",
            "infeasible"]
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
    periods = 8
    soliter = 0
    model = stnModelRobust()
    while not (len(model.m_list) == periods):
        model = stnModelRobust()
        stn = model.stn

        # states
        stn.state('F1',     init=2000000)   # Feed F1
        stn.state('F2',     init=2000000)   # Feed F2
        stn.state('P1', price=10, scost=9)  # Product P1
        stn.state('P2', price=20, scost=5)  # Product P2
        stn.state('I1', scost=15)  # Product P2

        # state to task arcs
        stn.stArc('F1',   'Heating')
        stn.stArc('F2',   'Reaction_1', rho=0.6)
        stn.stArc('I1', 'Reaction_1', rho=0.4)
        stn.stArc('P1', 'Reaction_2', rho=0.3)
        stn.stArc('I1', 'Reaction_2', rho=0.7)

        # task to state arcs
        stn.tsArc('Heating',    'I1')
        stn.tsArc('Reaction_1',    'P1')
        stn.tsArc('Reaction_2',    'P2')

        # unit-task data
        stn.unit('Heater', 'Heating', Bmin=40, Bmax=100, tm=5, rmax=80,
                 rinit=43, a=600, b=300)
        stn.unit('Reactor', 'Reaction_1', Bmin=30, Bmax=140, tm=5, rmax=120,
                 rinit=72, a=600, b=300)
        stn.unit('Reactor', 'Reaction_2', Bmin=30, Bmax=140, tm=5, rmax=120,
                 rinit=72, a=600, b=300)

        # operating mode and degradation data
        stn.opmode('Slow')
        stn.opmode('Normal')
        stn.ijkdata('Heating', 'Heater', 'Slow', 9, 5)
        stn.ijkdata('Heating', 'Heater', 'Normal', 6, 11)
        stn.ijkdata('Reaction_1', 'Reactor', 'Slow', 6, 7)
        stn.ijkdata('Reaction_1', 'Reactor', 'Normal', 4, 9)
        stn.ijkdata('Reaction_2', 'Reactor', 'Slow', 10, 5)
        stn.ijkdata('Reaction_2', 'Reactor', 'Normal', 6, 13)

        # time horizons
        Ts = 30
        dTs = 1
        Tp = 30*8
        dTp = 30
        TIMEp = range(0, Tp, dTp)

        # demand for Product P
        # demand_1 = [150, 88, 125, 67, 166]
        demand_1 = [150, 88, 125, 167, 166, 125, 302, 94, 300, 300, 300, 300]
        demand_2 = [50, 188, 131, 27, 141, 155, 122, 104, 300, 300, 300, 300]
        # demand_1 = np.random.uniform(50, 200, 8)
        # demand_2 = np.random.uniform(50, 200, 8)
        # model.demand('P1', Ts-dTs, demand_1[0])
        # model.demand('P2', Ts-dTs, demand_2[0])

        for i in range(0, len(TIMEp)):
            model.demand('P1', TIMEp[i], demand_1[i])
            model.demand('P2', TIMEp[i], demand_2[i])
        eps = 1 - sct.norm.ppf(q=q, loc=1, scale=0.27)
        model.uncertainty(eps)
        model.solve([Ts, dTs, Tp, dTp],
                    solver="cplex",
                    objective="terminal",
                    decisionrule="continuous",
                    periods=periods,
                    prefix="toy2R{0:.2f}".format(q),
                    tindexed=False,
                    save=True,
                    trace=True,
                    rdir=rdir)
        soliter += 1
        if soliter > 1:
            break
    profile = model.get_unit_profile("Reactor")
    profile["id"] = rid
    dfp = dfp.append(profile)
    print("MCS Heater")
    pheater = 0  # deg.simulate_deg(20000, model, "Heater", Sinit=43)
    print("MCS Reactor")
    preactor = deg.simulate_deg(20000, model, "Reactor",
                                Sinit=72)
    infeasible = soliter > 1
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
    line = [rid, q, eps, pheater, preactor, cost_storage,
            cost_maint, cost, cost0, dslack,
            ttot, infeasible]
    df.loc[rid] = line
    print(df)
# df = df.append(df2)
    df.to_pickle(rdir+"/results.pkl")
    df.to_csv(rdir+"/results.csv")
    dfp = dfp.append(dfp2)
    dfp.to_pickle(rdir+"/profile.pkl")
    dfp.to_csv(rdir+"/profile.csv")
    eps_target = 10

    return abs(preactor - eps_target)/(100-eps_target)*2 - 1


if __name__ == "__main__":

    # bo = BayesianOptimization(target, {'q': (0.05, 0.5)})
    # bo.maximize(init_points=10, n_iter=30, acq='ei', xi=0.01)
    bo = gp_minimize(target, [(0.05, 0.5)], acq_func="EI", n_calls=15,
                     n_random_starts=5, noise=0.1)
    import ipdb; ipdb.set_trace()  # noqa
