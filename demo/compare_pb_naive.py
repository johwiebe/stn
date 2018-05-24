#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dill
import sys
sys.path.append("../STN/modules")
import deg  # noqa

if __name__ == "__main__":
    with open("../data/model.pkl", "rb") as dill_file:
        m = dill.load(dill_file)
    TPfile = "../data/TP.pkl"
    Nmc = 200
    N = 1000
    # for j in ["Reactor_1"]:  #, "Reactor_2", "Heater", "Still"]:
    for j in ["Reactor_1", "Reactor_2", "Heater", "Still"]:
        print(j)
        p1 = deg.calc_p_fail(m, j, 0.3, TPfile, N, Nmc, pb=True, dt=3,
                             periods=12
                             )
        p2 = deg.calc_p_fail(m, j, 0.3, TPfile, N, Nmc, pb=False, dt=3,
                             periods=12
                             )
        print(max(p1), max(p2))
