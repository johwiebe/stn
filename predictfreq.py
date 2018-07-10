#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict frequencies from logistic regression
"""

import dill
import sys
import pyDOE
import numpy as np
import pandas as pd
sys.path.append('../STN/modules')


with open("../data/p4.dat", "rb") as dill_file:
    stn = dill.load(dill_file)

mlhs = pyDOE.lhs(len(stn.products), samples=500, criterion="maximin")*400 + 100
df = pd.DataFrame(mlhs, columns=stn.products)

with open("../data/p4freq.pkl", "rb") as f:
    lr = dill.load(f)
# j = "Heater"
# tms = ["Heating-Slow", "Heating-Normal"]
# for tm in tms:
#     lrtm = lr[j, tm]
#     probas = lrtm.predict_proba(mlhs)
#     classes = lrtm.classes_.astype(float)
#     import ipdb; ipdb.set_trace()  # noqa
#     meanfreq = probas * classes
#     meanfreq = meanfreq.sum(axis=1)
#     pred = classes[np.argmax(probas, axis=1)]
#     df[tm + "-mean"] = meanfreq
#     df[tm + "-max"] = pred
# df.to_pickle("../results/toypreds.pkl")
# df.to_csv("../results/toypreds.csv")

j = "R1"
tms = ["T1-Slow", "T1-Normal", "T1-Fast", "T2-Slow",
       "T2-Normal", "T2-Fast"]
for tm in tms:
    lrtm = lr[j, tm]
    probas = lrtm.predict_proba(mlhs)
    classes = lrtm.classes_.astype(float)
    meanfreq = probas * classes
    meanfreq = meanfreq.sum(axis=1)
    pred = classes[np.argmax(probas, axis=1)]
    df[tm + "-mean"] = meanfreq
    df[tm + "-max"] = pred
df.to_pickle("../results/p4predsR1.pkl")
df.to_csv("../results/p4predsR1.csv")
