#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train logistic regression for Markov-chain or frequency approach.
"""

import dill
import collections
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import stn.deg as deg  # noqa
from stn import blockPlanning  # noqa


def get_log_reg_list(prof, stn):
    """
    Train logistic regression (Markov-chain approach) for each task-mode
    combination.
        prof: task-mode data generated using lhs.py
        stn: stn structure
    """
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
    """
    Train logistic regression (Frequency approach) for each task-mode
    combination.
        prof: task-mode data generated using lhs.py
        stn: stn structure
    """
    TP = {}
    prods = stn.products
    for j in stn.units:
        for i in stn.I[j]:
            for k in stn.O[j]:
                tm = i + "-" + k
                TP[j, tm] = get_logreg_freq(prof, tm, j, prods)
    return TP


def get_logreg(prof, tm, j, prods):
    """
    Train logistic regression (Markov-chain approach).
        prof: task-mode data generated using lhs.py
        tm: task-mode
        j: name of unit
        prods: list of products
    """
    # Filter relevant data
    dfj = prof.loc[prof["unit"] == j, ].copy()
    dfj["tm"] = [row["task"] + "-" + row["mode"] for i, row in dfj.iterrows()]
    dfj["tm-1"] = dfj["tm"].shift(-1)
    dfj.loc[pd.isna(dfj["tm-1"]), "tm-1"] = "None-None"
    dfj = dfj[dfj["tm"] == tm]
    # Train logistic regression
    if dfj.shape[0] > 0 and len(np.unique(dfj["tm-1"])) > 1:
        X = np.array(dfj[prods])
        Y = np.array(dfj["tm-1"])
        if(len(np.unique(Y)) > 2):
            # Multinomial if more than 2 classes
            logreg = linear_model.LogisticRegression(multi_class="multinomial",
                                                     solver="lbfgs",
                                                     # solver="sag",
                                                     max_iter=10000,
                                                     verbose=2)
        else:
            # Binomial if only two classes
            logreg = linear_model.LogisticRegression(max_iter=10000,
                                                     verbose=2)
        logreg.fit(X, Y)
        return logreg
    elif dfj.shape[0] > 0:
        return np.array(dfj["tm-1"])[0]
    else:
        return "None-None"


def get_logreg_freq(prof, tm, j, prods):
    """
    Train logistic regression (Frequency approach).
        prof: task-mode data generated using lhs.py
        tm: task-mode
        j: name of unit
        prods: list of products
    """
    # unit specific data
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
    # TODO: these should be arguments to the script
    prof_file = '/home/jw3617/STN/results_p6/lhs/profile.pkl'  # From lhs.py
    stn_file = 'data/p6.dat'
    # Read task-mode data
    prof = pd.read_pickle(prof_file)
    # Load STN structure
    with open(stn_file, "rb") as dill_file:
        stn = dill.load(dill_file)
    # Train logistic regression Frequency approach
    TP = get_log_reg_list_freq(prof, stn)
    with open("data/p6freq.pkl", "wb") as dill_file:
        dill.dump(TP, dill_file)
    # Train logistic regression Markov-chain approach
    TP = get_log_reg_list(prof, stn)
    with open("data/p6mc.pkl", "wb") as dill_file:
        dill.dump(TP, dill_file)
