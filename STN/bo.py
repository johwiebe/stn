#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import yaml
import dill
import pandas as pd
from skopt import gp_minimize
sys.path.append('../STN/modules')
from stn import stnModel, stnModelRobust # noqa
import deg  # noqa


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

    return float(obj)


if __name__ == "__main__":
    global y
    with open(sys.argv[1], "r") as f:
        y = yaml.load(f)

    bo = gp_minimize(target, [(0.05, 0.5)], acq_func="EI", n_calls=20,
                     n_random_starts=5)  # , noise=0.2)
    bo_x = [x[0] for x in bo.x_iters]
    bo_y = list(bo.func_vals)
    df = pd.DataFrame([list(i) for i in zip(bo_x, bo_y)],
                      columns=["alpha", "cost"])
    df.to_pickle(y["rdir"] + "/bo.pkl")
    df.to_csv(y["rdir"] + "/bo.csv")

    with open(y["rdir"] + "/gp.pkl", "wb") as f:
        dill.dump(bo, f)
