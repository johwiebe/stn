#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate performance metrics logistic regression.
    rdir: directory with results
    stn: STN structure data file
"""

import dill
import argparse
import numpy as np
import pandas as pd
from stn import blockPlanning  # noqa
import stn.deg as deg  # noqa


def calc_metrics(rdir, stn, scenario, bound, j):
    """
    Calculate metrics.
        rdir: results directory
        stn: stn structure
        scenario: demand scenario
        bound: mc (Markov-chain) or freq (Frequency)
        j: unit
    """
    fname = {"mc": "_mc_results.pkl", "freq": "_freq_results.pkl"}
    print(rdir+"/det/"+scenario+"_results.pkl")
    # Load results
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
    rms_max = np.sqrt(sum((detmax[j] - detmax["mc"]) ** 2)/detmax.shape[0])
    p_out = sum(det[j] > det["mc"]) / N * 100
    detout = det[det[j] > det["mc"]]
    rms_out = 0
    if detout.shape[0] > 0:
        rms_out = np.sqrt(sum((detout[j] - detout["mc"]) ** 2)/detout.shape[0])
    detweird = det
    detweird["val"] = detweird[j] - detweird["mc"]
    detweird.loc[detweird["val"] < 0, "val"] = 0
    rms_weird = np.sqrt(sum(detweird["val"])/N)
    return [j, scenario, bound, rms_all, rms_max, p_out, rms_out, rms_weird]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rdir", help="directory with results")
    parser.add_argument("stn", help="stn data file")
    args = parser.parse_args()
    with open(args.stn, "rb") as f:
        stn = dill.load(f)
    scenarios = ["low", "avg", "high"]
    bounds = ["mc", "freq"]
    fname = {"mc": "_mc_results.pkl", "freq": "_freq_results.pkl"}
    d = [calc_metrics(args.rdir, stn, s, b, j)
         for j in stn.units
         for s in scenarios
         for b in bounds]
    df = pd.DataFrame(d, columns=["unit", "scenario", "bound", "rms_all",
                                  "rms_max", "p_out", "rms_out", "rms_weird"])
    df.to_pickle(args.rdir + "/metrics.pkl")
    df.to_csv(args.rdir + "/metrics.csv")
    print(df)
