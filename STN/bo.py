#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  21 08:37:40 2018

@author: johannes
"""

import sys
import yaml
import dill
import pandas as pd
from skopt import gp_minimize
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../STN/modules')
import deg  # noqa
from stn import stnModel, stnModelRobust # noqa


def target(x):
    q = x[0]

    TIMEp = range(0, y["Tp"], y["dTp"])

    with open(y["stn"], "rb") as dill_file:
        stn = dill.load(dill_file)

    if y["robust"]:
        model = stnModelRobust(stn)
    else:
        model = stnModel(stn)
    for i, t in enumerate(TIMEp):
        for p in stn.products:
            model.demand(p, t, y[p][i])

    model.solve([y["Ts"], y["dTs"], y["Tp"], y["dTp"]],
                solver="cplex",
                objective="terminal",
                periods=1,
                prefix=y["prfx"],
                rdir=y["rdir"],
                save=True,
                alpha=q,
                trace=True,
                solverparams=y["solverparams"],
                tindexed=False)
    df = model.eval(periods=y["periods"], TP=y["TP"])
    obj = df["Cost"]
    for j in stn.units:
        obj += df[j]/100*y["ccm"][j]

    return scaler.transform(float(obj))[0][0]


if __name__ == "__main__":
    global y, scaler
    with open(sys.argv[1], "r") as f:
        y = yaml.load(f)

    scaler = MinMaxScaler()
    scaler.fit([[0], [1]])

    x_init = [[0.02], [0.1], [0.2], [0.3], [0.4], [0.5]]
    y_init = [[target(x)] for x in x_init]
    scaler.fit(y_init)
    y_init = [yi[0] for yi in scaler.transform(y_init)]
    N = 15

    bo = gp_minimize(target, [(0.02, 0.5)], x0=x_init, y0=y_init,
                     acq_func="EI", n_calls=N, verbose=True,
                     n_random_starts=0, noise="gaussian", n_jobs=-1)
    bo_x = [x[0] for x in bo.x_iters]
    bo_y = list(bo.func_vals)
    df = pd.DataFrame([list(i) for i in zip(bo_x, bo_y)],
                      columns=["alpha", "cost"])
    df.to_pickle(y["rdir"] + "/bo.pkl")
    df.to_csv(y["rdir"] + "/bo.csv")

    with open(y["rdir"] + "/gp.pkl", "wb") as f:
        dill.dump(bo, f)
