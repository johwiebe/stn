#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dill
import collections
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from stn import deg  # noqa
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


def get_log_reg_list_freq(prof, stn):
    TP = {}
    prods = stn.products
    for j in stn.units:
        for i in stn.I[j]:
            for k in stn.O[j]:
                tm = i + "-" + k
                TP[j, tm] = get_logreg_freq(prof, tm, j, prods)
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
                                                     solver="lbfgs",
                                                     # solver="sag",
                                                     max_iter=10000,
                                                     verbose=2)
        else:
            logreg = linear_model.LogisticRegression(max_iter=10000,
                                                     verbose=2)
        logreg.fit(X, Y)
        return logreg
    elif dfj.shape[0] > 0:
        return np.array(dfj["tm-1"])[0]
    else:
        return "None-None"


def score(TP, df, prods, stn, alpha, dTs, dTp):
    scr = 0
    print("Calc score")
    df["taskmode"] = df["task"] + "-" + df["mode"]
    for j in stn.units:
        dfj = df[df["unit"] == j].copy()
        dfj = dfj.reset_index()
        for rid in np.unique(dfj["id"]):
            dfrid = dfj[dfj["id"] == rid]
            dem = [[d] for d in dfrid.loc[dfrid.index[0], prods].tolist()]
            hist_pred = deg.calc_p_fail_dem(dem, stn_file, j, alpha, TP=TP,
                                            dTs=dTs, dTp=dTp)
            c = collections.Counter(dfj.loc[dfj["id"] == rid, "taskmode"])
            hist_true = {tm: c[tm] for tm in hist_pred}
            scr += sum(np.array([(hist_true[tm] - hist_pred[tm])**2 for tm in
                                 hist_true]))
        print(scr)
    return scr


def cv(df, stn, alpha, dTs, dTp):
    ids = np.unique(df["id"])
    kf = KFold(n_splits=10)
    for k, (train, test) in enumerate(kf.split(ids)):
        df_train = df[df["id"] in train]
        df_test = df[df["id"] in test]
        TP = get_log_reg_list(df_train, stn)
        scr = score(TP, df_test, stn.products, stn, alpha, dTs, dTp)
    return scr


def get_logreg_freq(prof, tm, j, prods):
    # unit specific data frame
    dfj = prof[prof["unit"] == j].copy()
    dfj = dfj.reset_index(drop=True)
    dfj["taskmode"] = dfj["task"] + "-" + dfj["mode"]
    # data frame with taskmode counts
    dfhist = dfj.groupby("id")["taskmode"].value_counts().unstack().fillna(0)
    dfhist[stn.products] = dfj.groupby("id")[stn.products].mean()
    # train log reg if taskmode is in data
    if tm in dfhist:
        X = dfhist[stn.products]
        y = np.array(dfhist[tm])
        y = y.astype(str)
        # multinomial if there are more than two outcomes
        if len(np.unique(y)) > 2:
            logreg = linear_model.LogisticRegression(C=1,
                                                     multi_class="multinomial",
                                                     solver="lbfgs",
                                                     verbose=2,
                                                     max_iter=1000000)
        # binomial otherwise
        else:
            logreg = linear_model.LogisticRegression(C=1, max_iter=1000000)
        # fit model
        logreg.fit(X, y)

        return logreg
    else:
        return None


if __name__ == '__main__':
    prof_file = '/home/jw3617/STN/results_p4/lhs/profile.pkl'
    stn_file = "data/p4.dat"
    prof = pd.read_pickle(prof_file)
    with open(stn_file, "rb") as dill_file:
        stn = dill.load(dill_file)
    TP = get_log_reg_list_freq(prof, stn)
    with open("data/p4freq.pkl", "wb") as dill_file:
        dill.dump(TP, dill_file)
    TP = get_log_reg_list(prof, stn)
    with open("data/p4mc.pkl", "wb") as dill_file:
        dill.dump(TP, dill_file)
