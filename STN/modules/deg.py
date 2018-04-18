#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate random degradation signals
"""
import numpy as np
import pandas as pd
import time
import dill
from joblib import Parallel, delayed
import scipy.stats as sct

class degradationModel(object):
    def __init__(self, unit, dist="normal"):
        valid_dists = ["normal"]
        assert dist in valid_dists, "Not a valid distribution: %s" & dist
        self.dist = dist
        self.unit = unit
        # dictionaries indexed by k
        self.mu = {}
        self.sd = {}

    def set_op_mode(self, k, mu, sd):
        self.mu[k] = mu
        self.sd[k] = sd

    def get_quantile(self, alpha, k, p):
        if self.dist == "normal":
            mu = self.mu[k]*p
            sd = self.sd[k]*np.sqrt(p)
            return sct.norm.ppf(q=alpha, loc=mu, scale=sd)


def simulate_wiener(Darr, j, N=1,
                    Sinit=0, S0=0, Rmax=0, plot=False):
    np.random.seed()
    Ns = Darr.shape[1]
    Ninf = 0
    S = np.ones(N)*Sinit
    for s in range(0, Ns):
        if Darr[2, s] < 0.5:
            dS = np.random.normal(loc=Darr[0, s], scale=Darr[1, s], size=N)
            S = S + dS
        else:
            S = np.ones(N)*S0
        Ninf += sum(S >= Rmax)
        S = S[S < Rmax]
        N = S.size
    return Ninf

def get_short_mc(mc, t):
    slast = mc[0]
    mcshort = [mc[0]]
    tshort = []
    for i, s in enumerate(mc):
        if s != slast:
            mcshort.append(s)
            tshort.append(t[i-1])
        slast = s
    tshort.append(t[-1])
    return mcshort, tshort

def get_gradient(stn, j):
    GL = {}
    LL = {}
    for i in stn.I[j]:
        for k in stn.O[j]:
            taskmode = i + "-" + k
            GL[taskmode] = stn.D[i, j, k]/stn.p[i, j, k]
            LL[taskmode] = (0.27*stn.D[i, j, k])**2/stn.p[i, j, k]
    GL["None-None"] = 0
    LL["None-None"] = 0.05**2
    GL["M-M"] = 0
    LL["M-M"] = 0.05**2
    return GL, LL

def gen_group(mc, t, sep):
    mcg = []
    tg = []
    for i, el in enumerate(mc):
        if el == sep:
            yield mcg, tg
            mcg = []
            tg = []
        mcg.append(el)
        tg.append(t[i])
    yield mcg, tg

def simulate_wiener_group(mc, t, GL, LL, Nmcs=1000, Sinit=0, Smax=0):
    np.random.seed()
    Dm = [GL[tm] for tm in mc]
    Dsd = [LL[tm] for tm in mc]
    tdiff = [t[0]]
    tdiff += [t - s for s, t in zip(t, t[1:])]
    c = (Smax - Sinit - np.cumsum(np.multiply(tdiff, Dm)))
    c = np.insert(c, 0, Smax - Sinit)
    N = len(tdiff)
    Dsqrt = np.diag(np.sqrt(tdiff))
    M = np.tril(np.ones((N, N), dtype=int), 0)
    hl = []
    for n in range(0, Nmcs):
        u = np.random.normal(scale=np.sqrt(Dsd))
        A = M @ Dsqrt @ u
        xp = c[1:] + A
        xm = np.insert(xp, 0, c[0])[:-1]
        ind = [xi > 0 for xi in xp]
        h = 1
        for i in range(0, N):
            if ind[i]:
                h *= (1 - np.exp(-2*xm[i]*xp[i]/(Dsd[i]*tdiff[i])))
            else:
                h = 0
        hl.append(h)
    return 1 - np.mean(hl)

def simulate_wiener_pb(mc, t, GL, LL, Nmcs=1000, Sinit=0, S0=0, Smax=0):
    J = 1
    # split sequence at maintenance tasks
    for mcg, tg in gen_group(mc, t, "M-M"):
        J *= 1 - simulate_wiener_group(mcg, tg, GL, LL, Nmcs, Sinit=Sinit, Smax=Smax)
        Sinit = S0
    return 1 - J

def simulate_deg_pb(N, Nmc, model, j, eps, dt=1/10, periods=0, pb=False, *args, **kwargs):
    Ncpus = 8
    # make data global for parallel execution
    global stn, table
    stn = model.stn
    # get schedules from model scheduling horizon
    df = model.get_unit_profile(j, full=False)
    df["taskmode"] = df["task"] + "-" + df["mode"]
    mc0 = list(df["taskmode"])
    t0 = list(df["time"])[1:]
    # length of final task in scheduling horizon
    i = df.tail(1)["task"].iloc[0]
    if i == "None":
        t0.append(t0[-1] + model.sb.dT)
    elif i == "M":
        t0.append(t0[-1] + stn.tau[j])
    else:
        k = df.tail(1)["mode"].iloc[0]
        t0.append(t0[-1] + stn.p[i, j, k])
    # load logistic regression model
    with open("TP.pkl", "rb") as dill_file:
        TP = dill.load(dill_file)
    # get production targets for planning horizon
    pdf = model.get_production_targets()
    if periods > 0:
        pdf = pdf[(pdf["time"] <= periods*model.pb.dT) & (pdf["time"] > t0[-1])]
    else:
        pdf = pdf[(pdf["time"] > t0[-1])]
    prods = [p for p in pdf.columns[pdf.columns != "time"]]
    prods = ["Product_1", "Product_2"]
    dem = []
    for p in prods:
        dem.append(np.array(pdf[p]))
    # generate Nmc sequences from Markov chain
    st = time.time()
    mclist = []
    mcslist = []
    tlist = []
    tslist = []
    # TODO: Change definition of R to make them compatible?
    if "Robust" in type(model).__name__:
        Sinit = model.model.sb.R[j, model.sb.T - model.sb.dT]()
    else:
        Sinit = model.stn.Rmax[j] - model.model.sb.R[j, model.sb.T - model.sb.dT]()
    D = {"None-None":0, "M-M":0}
    # calculate all relavent transition probabilities once
    table = {}
    for i in stn.I[j]:
        for k in stn.O[j]:
            tm = i + "-" + k
            D.update({tm:stn.D[i,j,k]*(1+eps)})
    for tm in D.keys():
        logreg = TP[j, tm]
        for period, d in enumerate(dem[0]):
            if type(logreg) == str:
                table[tm, period] = pd.DataFrame([1], columns=[logreg])
            else:
                table[tm, period] = np.cumsum(pd.DataFrame(logreg.predict_proba([[d[period] for d in dem]]),
                                     columns=logreg.classes_), axis=1)
    # generate sequences in parallel
    res = Parallel(n_jobs=Ncpus)(delayed(simulate_mc)(D,
                                                       j, "None-None",
                                                       t0[-1],
                                                       model.sb.dT,
                                                       model.pb.dT,
                                                       dem,
                                                       eps,
                                                       Sinit=Sinit)
                                  for i in range(0, Nmc))
    # append generated sequences to scheduling horizon
    for n in range(0, Nmc):
        mc = mc0 + res[n][0]
        t = t0 + res[n][2]
        mcshort, tshort = get_short_mc(mc, t)
        mclist.append(mc)
        tlist.append(t)
        mcslist.append(mcshort)
        tslist.append(tshort)
    # estimate failure probabilities in parallel
    Smax = model.stn.Rmax[j]
    Sinit = model.stn.Rinit[j]
    print(np.mean([len(mc) for mc in mcslist]))
    # approach by Poetzelberger
    if pb:
        GL, LL = get_gradient(stn, j)
        inflist = Parallel(n_jobs=Ncpus)(delayed(simulate_wiener_pb)(mcslist[i],
                                                                     tslist[i],
                                                                     GL, LL,
                                                                     Nmcs=N,
                                                                     Smax=Smax,
                                                                     Sinit=Sinit,
                                                                     *args,
                                                                     **kwargs)
                                         for i in range(0, len(mcslist)))
        inflist = np.array(inflist)*100
    # naive approach
    else:
        Darrlist = []
        for n in range(0, Nmc):
            Darrlist.append(get_deg_profile(mclist[n], stn, j, model.sb.dT, dt,
                                            Sinit=Sinit))
        inflist = Parallel(n_jobs=Ncpus)(delayed(simulate_wiener)(Darr, j,
                                                                  N=N,
                                                                  Rmax=Smax,
                                                                  Sinit=Sinit,
                                                                  *args,
                                                                  **kwargs)
                                         for Darr in Darrlist)
        inflist = np.array(inflist)/N*100

    print("Time taken:"+str(time.time()-st))

    return inflist

def simulate_mc(D, j, s0, t0, dTs, dTp, demand, eps, Sinit=0):
    np.random.seed()
    mc = []
    s = s0
    Smax = stn.Rmax[j]
    S = Sinit
    Slist = []
    t = t0
    # time taken by s0
    if s == "None-None":
        t += dTs
    elif s == "M-M":
        t += stn.tau[j]
    else:
        i, k = s.split("-")
        t += stn.p[i, j, k]
    tlist = []
    # add operating modes to while t < T
    while t < (t0 // dTp + len(demand[0]))*dTp:
        mc.append(s)
        Slist.append(S)
        tlist.append(t)
        # TODO: this should not be necessary, MC should not contain maintenance
        while True:
            # draw random new state from transition probabilities
            s_ind = np.where(np.random.uniform() < (table[s, t // dTp - t0 // dTp]))[1][0]
            s = table[s, t // dTp - t0 // dTp].columns[s_ind]
            if s != "M-M":
                break
        S = S + D[s]
        # insert maintenance if needed
        if S > Smax:
            s = "M-M"
            S = 0
        if s == "None-None":
            t += dTs
        elif s == "M-M":
            t += stn.tau[j]
        else:
            i, k = s.split("-")
            t += stn.p[i, j, k]
    return mc, Slist, tlist

def get_deg_profile(profile, stn, j, dT, dt=1/10, N=1, Sinit=0, S0=0):
    Ns = int(len(profile)*dT/dt)
    Darr = np.zeros((3, 0))
    t = 0
    for taskmode in profile:
        m = 0
        mue = 0
        sd = 0.05*np.sqrt(dt)
        if taskmode == "None-None":
            tend = t + dT
        elif taskmode == "M-M":
            tend = t + stn.tau[j]
            sd = 0
            m = 1
        else:
            s = taskmode.split("-")
            i = s[0]
            k = s[1]
            tend = t + stn.p[i,j,k]
            mue = stn.D[i,j,k]*dt/stn.p[i,j,k]
            sd = 0.27*mue/(np.sqrt(dt/stn.p[i,j,k]))
        tend = int(tend)
        np.array([[1, 2],[3,4]])
        Darr = np.concatenate((Darr,
                        np.array([
                            [mue for i in range(int(t/dt), int(tend/dt))],
                            [sd for i in range(int(t/dt), int(tend/dt))],
                            [m]+[0 for i in range(int(t/dt)+1, int(tend/dt))]])),
                       axis=1)
        t = tend
    return Darr
