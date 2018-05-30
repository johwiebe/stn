#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  21 08:37:40 2018

@author: johannes
"""

import sys
import yaml
import dill
import functools
import argparse
import pandas as pd
from skopt import gp_minimize
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../STN/modules')
import deg  # noqa
from stn import stnModel, stnModelRobust # noqa


def target(y, scaler, x):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help=".yaml file with run parameters")
    parser.add_argument("prefix", help="prefix for file names")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        y = yaml.load(f)
    y["prfx"] = args.prefix + y["prfx"]

    scaler = MinMaxScaler()
    scaler.fit([[0], [1]])

    wrap = functools.partial(target, y, scaler)

    x_init = [[0.02], [0.17], [0.34], [0.5]]
    y_init = [[wrap(x)] for x in x_init]
    scaler.fit(y_init)
    y_init = [yi[0] for yi in scaler.transform(y_init)]
    N = 34

    bo = gp_minimize(wrap, [(0.02, 0.5)], x0=x_init, y0=y_init,
                     acq_func="EI", n_calls=N, verbose=True,
                     n_random_starts=0, noise="gaussian", n_jobs=-1)
    bo_x = [x[0] for x in bo.x_iters]
    bo_y = scaler.inverse_transform(bo.func_vals.reshape(-1, 1))
    bo_y = [yi[0] for yi in bo_y]
    df = pd.DataFrame([list(i) for i in zip(bo_x, bo_y)],
                      columns=["alpha", "cost"])
    df.to_pickle(y["rdir"] + "/" + y["prfx"] + "obj.pkl")
    df.to_csv(y["rdir"] + "/" + y["prfx"] + "obj.csv")

    with open(y["rdir"] + "/" + y["prfx"] + "bo.pkl", "wb") as f:
        dill.dump(bo, f)
