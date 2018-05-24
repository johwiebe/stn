#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import sys
import dill
import copy
sys.path.append('../STN/modules')
from blocks import (blockSchedulingRobust, blockScheduling, blockPlanning,
                    blockPlanningRobust) # noqa
from stn import stnModel, stnModelRobust # noqa

demand_1 = [150, 88, 125, 67, 166, 203, 90, 224, 174, 126, 66, 119, 234, 64,
            103, 77, 132, 186, 174, 239, 124, 194, 91, 228]
demand_2 = [200, 150, 197, 296, 191, 193, 214, 294, 247, 313, 226, 121, 197,
            242, 220, 342, 355, 320, 335, 298, 252, 222, 324, 337]
obj = "terminal"
dr = "continuous"
a = 0.5

Ts = 168
dTs = 3
Tp = 168*24
dTp = 168
TIMEs = range(0, Ts, dTs)
TIMEp = range(0, Tp, dTp)

solverparams = {
                "timelimit": 60,
                "mipgap": 0.02,
               }
import ipdb; ipdb.set_trace()  # noqa
rdir = "/home/jw3617/STN/results"
with open("../data/biondiD.dat", "rb") as dill_file:
    stnd = dill.load(dill_file)
with open("../data/biondiR.dat", "rb") as dill_file:
    stnr = dill.load(dill_file)
# scheduling - deterministic
# model = blockScheduling(copy.copy(stnd), [0, Ts, dTs],
#                         {"Product_1": demand_1[0],
#                          "Product_2": demand_2[0]})
# model.build(objective=obj, decisionrule=dr, alpha=a)
# model.solve(
#             solver="cplex",
#             prefix="biondi_sched_det",
#             rdir="/home/jw3617/STN/results",
#             tindexed=False,
#             save=True,
#             trace=True,
#             solverparams=solverparams)
# 
# # scheduling - robust
# model = blockSchedulingRobust(copy.copy(stnr), [0, Ts, dTs],
#                               {"Product_1": demand_1[0],
#                                "Product_2": demand_2[0]})
# model.build(objective=obj, decisionrule=dr, alpha=a)
# model.solve(
#             solver="cplex",
#             prefix="biondi_sched_rob",
#             rdir="/home/jw3617/STN/results",
#             tindexed=False,
#             save=True,
#             trace=True,
#             solverparams=solverparams)
# 
# # planning - deterministic
# with open("../data/biondiD.dat", "rb") as dill_file:
#     stnd = dill.load(dill_file)
# model = blockPlanning(copy.copy(stnd), [0, Tp, dTp],
#                       {"Product_1": 150, "Product_2": 200})
# for n, t in enumerate(TIMEp):
#     model.demand('Product_1', t, demand_1[n])
#     model.demand('Product_2', t, demand_2[n])
# 
# model.build(objective=obj, decisionrule=dr, alpha=a)
# 
# model.solve(
#             solver="cplex",
#             prefix="biondi_plan_det",
#             rdir="/home/jw3617/STN/results",
#             tindexed=False,
#             save=True,
#             trace=True,
#             solverparams=solverparams)
# 
# # planning - robust
# with open("../data/biondiR.dat", "rb") as dill_file:
#     stnr = dill.load(dill_file)
# model = blockPlanningRobust(copy.copy(stnr), [0, Tp, dTp],
#                             {"Product_1": 150, "Product_2": 200})
# for n, t in enumerate(TIMEp):
#     model.demand('Product_1', t, demand_1[n])
#     model.demand('Product_2', t, demand_2[n])
# 
# model.build(objective=obj, decisionrule=dr, alpha=a)
# 
# model.solve(
#             solver="cplex",
#             prefix="biondi_plan_rob",
#             rdir="/home/jw3617/STN/results",
#             tindexed=False,
#             save=True,
#             trace=True,
#             solverparams=solverparams)
# 
# # integrated - deterministic
# 
# solverparams = {"timelimit": 120,
#                 "mipgap": 0.01}
# 
# with open("../data/biondiR.dat", "rb") as dill_file:
#     stnd = dill.load(dill_file)
# model = stnModel(copy.copy(stnd))
# 
# for i in range(0, len(TIMEp)):
#     model.demand('Product_1', TIMEp[i], demand_1[i])
#     model.demand('Product_2', TIMEp[i], demand_2[i])
# 
# model.solve([Ts, dTs, Tp, dTp],
#             solver="cplex",
#             objective=obj,
#             decisionrule=dr,
#             # periods=12,
#             prefix="biondi_int_det",
#             rdir="/home/jw3617/STN/results",
#             tindexed=False,
#             trace=True,
#             save=True,
#             alpha=a,
#             solverparams=solverparams)
# ph = model.calc_p_fail("Heater", periods=12)
# import ipdb; ipdb.set_trace()  # noqa

# integrated - robust

with open("../data/biondiR.dat", "rb") as dill_file:
    stnr = dill.load(dill_file)
model = stnModelRobust(copy.copy(stnr))

for i in range(0, len(TIMEp)):
    model.demand('Product_1', TIMEp[i], demand_1[i])
    model.demand('Product_2', TIMEp[i], demand_2[i])

model.solve([Ts, dTs, Tp, dTp],
            solver="cplex",
            objective=obj,
            decisionrule=dr,
            # periods=12,
            prefix="biondi_int_rob",
            rdir="/home/jw3617/STN/results",
            tindexed=False,
            trace=True,
            save=True,
            alpha=a,
            solverparams=solverparams)
ph = model.calc_p_fail("Heater", periods=12)
import ipdb; ipdb.set_trace()  # noqa
