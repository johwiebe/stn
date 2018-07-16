#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize uncertainty set size using Bayesian Optimization.
    file: .dat file with stn structure
    prfx: prefix for file names.
"""

import yaml
import dill
import functools
import argparse
import pandas as pd
import stn.deg as deg  # noqa
from stn import stnModel, stnModelRobust # noqa
from skopt import gp_minimize
from sklearn.preprocessing import MinMaxScaler

def target(y, scaler, x):
    """
    Target function for Bayesian Optimization
        y: config dictionary (.yaml)
        scaler: scaler to normalize data
        x: alpha for which to evaluate
    """
    q = x[0]

    TIMEp = range(0, y["Tp"], y["dTp"])

    # Load stn structure
    with open(y["stn"], "rb") as dill_file:
        stn = dill.load(dill_file)
    # Initialize model
    if y["robust"]:
        model = stnModelRobust(stn)
    else:
        model = stnModel(stn)
    # Add demands
    for i, t in enumerate(TIMEp):
        for p in stn.products:
            model.demand(p, t, y[p][i])
    # Solve model
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
    # Evaluate overall cost
    df = model.eval(periods=y["periods"], TP=y["TP"])
    obj = df["Cost"]
    for j in stn.units:
        obj += df[j]/100*y["ccm"][j]
    # Return scaled cost
    return scaler.transform(float(obj))[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help=".yaml file with run parameters")
    parser.add_argument("prefix", help="prefix for file names")
    args = parser.parse_args()
    # Load config file
    with open(args.file, "r") as f:
        y = yaml.load(f)
    y["prfx"] = args.prefix + y["prfx"]
    # Initialize Scaler
    scaler = MinMaxScaler()
    scaler.fit([[0], [1]])
    # Single variable target function
    wrap = functools.partial(target, y, scaler)
    # Get initial points and scale
    x_init = [[0.02], [0.17], [0.34], [0.5]]
    y_init = [[wrap(x)] for x in x_init]
    scaler.fit(y_init)
    y_init = [yi[0] for yi in scaler.transform(y_init)]

    # Maximum number of iterations
    N = 34
    # Bayesian Optimization
    bo = gp_minimize(wrap, [(0.02, 0.5)], x0=x_init, y0=y_init,
                     acq_func="EI", n_calls=N, verbose=True,
                     n_random_starts=0, noise="gaussian", n_jobs=-1)
    # Unscale and save results
    bo_x = [x[0] for x in bo.x_iters]
    bo_y = scaler.inverse_transform(bo.func_vals.reshape(-1, 1))
    bo_y = [yi[0] for yi in bo_y]
    df = pd.DataFrame([list(i) for i in zip(bo_x, bo_y)],
                      columns=["alpha", "cost"])
    df.to_pickle(y["rdir"] + "/" + y["prfx"] + "obj.pkl")
    df.to_csv(y["rdir"] + "/" + y["prfx"] + "obj.csv")

    with open(y["rdir"] + "/" + y["prfx"] + "bo.pkl", "wb") as f:
        dill.dump(bo, f)
