#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:37:40 2017

@author: jeff
"""

import sys
import dill
import numpy as np
import pandas as pd
sys.path.append('../STN/modules')
from blocks import blockPlanning  # noqa
import deg  # noqa


def calc_metrics(rdir, stn, scenario, bound, j):
    fname = {"mc": "_results.pkl", "freq": "_freq_results.pkl"}
    print(rdir+"/det/"+scenario+"_results.pkl")
    with open(rdir+"/det/"+scenario+"_results.pkl", "rb") as f:
        det = dill.load(f)
        det = det.rename(index=str, columns={"ID": "id",
                                             "Preactor1": "Reactor_1",
                                             "Preactor2": "Reactor_2",
                                             "Pstill": "Still",
                                             "Pheater": "Heater"})
        det = det[list(stn.units) + ["alpha"]].reset_index(drop=True)
    with open(rdir+"/mc/"+scenario+fname[bound], "rb") as f:
        mc = dill.load(f)
        mc = mc[list(stn.units) + ["alpha"]].reset_index(drop=True)
    N = det.shape[0]
    det["mc"] = np.interp(det["alpha"], mc["alpha"], mc[j])
    rms_all = np.sqrt(sum((det[j] - det["mc"]) ** 2)/N)
    detmax = det.loc[det.groupby("alpha")[j].idxmax(),
                     [j, "alpha", "mc"]]
    rms_max = np.sqrt(sum((detmax[j] - detmax["mc"]) ** 2)/N)
    p_out = sum(det[j] > det["mc"]) / N * 100
    return [j, scenario, bound, rms_all, rms_max, p_out]


rdir = sys.argv[1]
stn_file = sys.argv[2]
with open(stn_file, "rb") as f:
    stn = dill.load(f)
scenarios = ["low", "avg", "high"]
bounds = ["mc", "freq"]
fname = {"mc": "_results.pkl", "freq": "_freq_results.pkl"}
d = [calc_metrics(rdir, stn, s, b, j)
     for j in stn.units
     for s in scenarios
     for b in bounds]
df = pd.DataFrame(d, columns=["unit", "scenario", "bound", "rms_all",
                              "rms_max", "p_out"])
df.to_pickle(rdir + "/metrics.pkl")
df.to_csv(rdir + "/metrics.csv")
print(df)
