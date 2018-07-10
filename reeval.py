#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import yaml
import dill
sys.path.append("../STN/modules")
from stn import blockPlanning  # noqa


with open(sys.argv[1], "r") as f:
    y = yaml.load(f)

prfx = y["rdir"] + "/" + y["prfx"]
res = pd.read_pickle(prfx + "results.pkl")
res = res.reset_index(drop=True)

TIMEp = range(0, y["Tp"], y["dTp"])

# res["alpha"] = y["alphas"]

for index, row in res.iterrows():
    with open(y["stn"], "rb") as dill_file:
        stn = dill.load(dill_file)

    model = blockPlanning(stn, [0, y["Tp"], y["dTp"]])
    for i in range(0, len(TIMEp)):
        for p in stn.products:
            model.demand(p, TIMEp[i], y[p][i])
    model.build(objective="terminal", decisionrule="continuous",
                alpha=row["alpha"],
                rdir=y["rdir"], prefix=y["prfx"]+sys.argv[2])

    model.loadres(prfx + str(row["id"]) + "STN.pyomo")
    df = model.calc_p_fail(TP=y["TP"], periods=12, Nmc=100, dTs=y["dTs"],
                           freq=y["freq"])
    for j in stn.units:
        res.loc[index, j] = max(df[j])

res.to_pickle(prfx+sys.argv[2]+"_results.pkl")
res.to_csv(prfx+sys.argv[2]+"_results.csv")
