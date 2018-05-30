#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find optimal alpha by random sampling."""

import argparse
import sys
import yaml
import dill
import functools
import pandas as pd
import numpy as np
sys.path.append('../STN/modules')
import deg  # noqa
from stn import stnModel, stnModelRobust # noqa


def target(y, x):
    """Target function to minimize."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help=".yaml file with run parameters")
    parser.add_argument("prefix", help="prefix for file names")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        y = yaml.load(f)
    y["prfx"] = args.prefix + y["prfx"]

    N = 40
    x0 = np.random.uniform(low=0.02, high=0.5, size=N)
    wrap = functools.partial(target, y)
    y0 = [wrap([x]) for x in x0]

    df = pd.DataFrame([list(i) for i in zip(x0, y0)],
                      columns=["alpha", "cost"])
    df.to_pickle(y["rdir"] + "/" + y["prfx"] + "obj.pkl")
    df.to_csv(y["rdir"] + "/" + y["prfx"] + "obj.csv")
