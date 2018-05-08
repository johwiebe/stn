#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import dill
import yaml
import pandas as pd
import numpy as np
from sklearn import linear_model
from collections import Counter
sys.path.append("modules")
import deg  # noqa
from stn import blockPlanning  # noqa


def get_log_reg_list(prof, stn):
    TP = {}
    prods = stn.products
    for j in stn.units:
        for i in stn.I[j]:
            for k in stn.O[j]:
                tm = i + "-" + k
                TP[j, tm] = get_logreg(prof, tm, j, prods)
        for tm in set(["None-None", "M-M"]):
            TP[j, tm] = get_logreg(prof, tm, j, prods)
    return TP


def get_logreg(prof, tm, j, prods):
    dfj = prof.loc[prof["unit"] == j, ].copy()
    dfj["tm"] = [row["task"] + "-" + row["mode"] for i, row in dfj.iterrows()]
    dfj["tm-1"] = dfj["tm"].shift(-1)
    dfj.loc[pd.isna(dfj["tm-1"]), "tm-1"] = "None-None"
    dfj = dfj[dfj["tm"] == tm]
    if dfj.shape[0] > 0 and len(np.unique(dfj["tm-1"])) > 1:
        X = np.array(dfj[prods])
        Y = np.array(dfj["tm-1"])
        if(len(np.unique(Y)) > 2):
            logreg = linear_model.LogisticRegression(multi_class="multinomial",
                                                     # solver="lbfgs",
                                                     solver="sag",
                                                     max_iter=10000,
                                                     verbose=0)
        else:
            logreg = linear_model.LogisticRegression(max_iter=10000,
                                                     verbose=0)
        logreg.fit(X, Y)
        return logreg
    elif dfj.shape[0] > 0:
        return np.array(dfj["tm-1"])[0]
    else:
        return "None-None"


# def get_log_reg(prof, tm, j):
#     dfj = prof.loc[prof["unit"] == j, ].copy()
#     dfj["tm-1"] = dfj["tm"].shift(-1)
#     dfj.loc[pd.isna["tm-1"], "tm-1"] = "None-None"
#     if np.any(dfj["tm"] == tm):
#         table = get_hist(dfj[dfj["tm"] == tm], "tm-1")
#     else:
#         table = get_hist(dfj[dfj["tm"] == "None-None"], "tm-1")
#     return table


# def get_hist(df, col):
#     table = pd.DataFrame.from_dict(Counter(df[col]), orient="index")
#     table = table.rename(columns={0: "count"})
#     table["p"] = table["count"]/sum(table["count"])
#     return table


if __name__ == "__main__":
    prof_file = "/home/jw3617/STN/results_p2/lhs/profile.pkl"
    stn_file = "../data/p2.dat"
    prof = pd.read_pickle(prof_file)
    with open(stn_file, "rb") as dill_file:
        stn = dill.load(dill_file)
    TP = get_log_reg_list(prof, stn)
    with open("../data/p2TP.pkl", "wb") as dill_file:
        dill.dump(TP, dill_file)
    # with open("../data/toy2TP.pkl", "rb") as dill_file:
    #     TP = dill.load(dill_file)

    # with open(sys.argv[1], "r") as f:
    #     y = yaml.load(f)
    # j = "Heater"
    # alpha = 0.5
    # model = blockPlanning(stn, [0, y["Tp"], y["dTp"]])
    # model.build(objective="terminal", decisionrule="continuous",
    #             alpha=alpha,
    #             rdir=y["rdir"], prefix=y["prfx"])
    # model.loadres(y["rdir"]+"/"+y["prfx"]
    #               + str(24)+"STN.pyomo")
    # df = model.get_production_targets()
    # import ipdb; ipdb.set_trace()
    # dem = [df["P1"].tolist(), df["P2"].tolist()]
    # # pdfloc = model.calc_p_fail(TP=TPfile, periods=12, Nmc=100)
    # deg.calc_p_fail_dem(dem, stn_file,
    #                     j, alpha,
    #                     TPfile="../data/toy2TP.pkl",
    #                     periods=12, dTs=1, dTp=30)
    # deg.calc_p_fail(model, j, alpha, y["TP"],
    #                 periods=12, dTs=3)
