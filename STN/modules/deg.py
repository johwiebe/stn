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
from math import floor


class degradationModel(object):
    """
    Degradation model for an STN unit.
        unit: name of unit
        dist: type of distribution
    """
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

    def get_quantile(self, alpha, k, dt=1):
        if self.dist == "normal":
            mu = self.mu[k]*dt
            sd = self.sd[k]*np.sqrt(dt)
            return sct.norm.ppf(q=alpha, loc=mu, scale=sd)

    def get_mu(self, k, dt=1):
        return self.mu[k]*dt

    def get_sd(self, k, dt=1):
        return self.sd[k]*np.sqrt(dt)

    def get_dist(self, k, dt=1):
        mu = self.mu[k]*dt
        sd = self.sd[k]*np.sqrt(dt)
        return mu, sd

    def get_eps(self, alpha, k, dt=1):
        mu = self.get_mu(k, dt=dt)
        eps = 1 - self.get_quantile(alpha, k, dt=dt)/mu
        return eps


def calc_p_fail(model, j, alpha, TPfile, Nmc=100, N=1000, dt=3,
                periods=0, pb=True, dTs=3, *args, **kwargs):
    """
    Calculate probability of unit failure
        model: solved stn model
        j: unit
        alpha: uncertainty set size parameter
        TPfile: file with logistic regression model for markov chain
        Nmc: number of sequences generated from markov chain
        N: Number of Monte-Carlo evaluations for each sequence
        dt: time step for naive approach
        periods: number of planning periods to evaluate (all if periods=0)
        pb: if set to True, approach by Poetzelberger is used (Wiener process)
    """
    Ncpus = 8  # number of CPUs to used for parallel execution
    # make data global for parallel execution
    global stn, table
    stn = model.stn
    # get schedules from model scheduling horizon
    if "get_unit_profile" in dir(model):
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
        Sinit = model.model.sb.R[j, model.sb.T - model.sb.dT]()
        dTp = model.pb.dT
        dTs = model.sb.dT
    else:
        mc0 = ["None-None"]
        t0 = [dTs]
        Sinit = stn.Rinit[j]
        dTp = model.dT
    #  load logistic regression model
    with open(TPfile, "rb") as dill_file:
        TP = dill.load(dill_file)
    # get production targets for planning horizon
    pdf = model.get_production_targets()
    if periods > 0:
        pdf = pdf[(pdf["time"] <= periods*dTp) & (pdf["time"]
                                                  > t0[-1])]
    else:
        pdf = pdf[(pdf["time"] > t0[-1])]
    prods = stn.products
    dem = []
    for p in prods:
        dem.append(np.array(pdf[p]))
    # generate Nmc sequences from Markov chain
    st = time.time()
    mclist = []
    mcslist = []
    tlist = []
    tslist = []
    D = {"None-None": 0, "M-M": 0}
    # calculate all relavent transition probabilities once
    table = {}
    for i in stn.I[j]:
        for k in stn.O[j]:
            tm = i + "-" + k
            ptm = stn.p[i, j, k]
            # Dtm = stn.D[i, j, k]
            # eps = 1 - stn.deg[j].get_quantile(alpha, tm, ptm)/Dtm
            Dtm = stn.deg[j].get_mu(tm, ptm)
            eps = stn.deg[j].get_eps(alpha, tm, ptm)
            D.update({tm: Dtm*(1+eps)})
    for tm in D.keys():
        logreg = TP[j, tm]
        for period, d in enumerate(dem[0]):
            if type(logreg) == str:
                table[tm, period] = pd.DataFrame([1], columns=[logreg])
            else:
                prob = logreg.predict_proba([[d[period] for d in dem]])
                table[tm, period] = np.cumsum(pd.DataFrame(prob,
                                              columns=logreg.classes_), axis=1)
    # generate sequences in parallel
    res = Parallel(n_jobs=Ncpus)(delayed(generate_seq_mc)(D,
                                                          j, "None-None",
                                                          t0[-1],
                                                          dTs, dTp,
                                                          dem,
                                                          # eps,
                                                          Sinit=Sinit)
                                 for i in range(0, Nmc))
    # append generated sequences to scheduling horizon
    # occ = []
    for n in range(0, Nmc):
        mc = mc0 + res[n][0]
        # occ.append(sum(np.array(mc) == "Separation-Slow"))
        t = t0 + res[n][2]
        mcshort, tshort = get_short_mc(mc, t)
        mclist.append(mc)
        tlist.append(t)
        mcslist.append(mcshort)
        tslist.append(tshort)
    # return occ
    # print(occ)
    # estimate failure probabilities in parallel
    Smax = model.stn.Rmax[j]
    Sinit = model.stn.Rinit0[j]
    # approach by Poetzelberger
    if pb:
        GL, LL = get_gradient(stn, j)
        inflist = Parallel(n_jobs=Ncpus)(delayed(sim_wiener_pb)(mcslist[i],
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
        inflist = Parallel(n_jobs=Ncpus)(delayed(sim_wiener_naive)(Darr, j,
                                                                   N=N,
                                                                   Rmax=Smax,
                                                                   Sinit=Sinit,
                                                                   *args,
                                                                   **kwargs)
                                         for Darr in Darrlist)
        inflist = np.array(inflist)/N*100

    print("Time taken:" + str(time.time()-st) + ", Pfail:" + str(max(inflist)))
    return inflist


def generate_seq_mc(D, j, s0, t0, dTs, dTp, demand, Sinit=0):
    """Generate sequence of operating modes from Marov chain."""
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
            s_ind = np.where(np.random.uniform()
                             < (table[s, t // dTp - t0 // dTp]))[1][0]
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


def sim_wiener_naive(Darr, j, N=1, Sinit=0, S0=0,
                     Rmax=0, plot=False):
    """Calculate probability of failure with naive approach."""
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


def sim_wiener_pb(mc, t, GL, LL, Nmcs=1000, Sinit=0, S0=0, Smax=0):
    """Calcualte probabillity of failure with approach by Poetzelberger."""
    J = 1
    # split sequence at maintenance tasks
    for mcg, tg in gen_group(mc, t, "M-M"):
        if len(mcg) > 0:
            J *= 1 - sim_wiener_group(mcg, tg, GL, LL, Nmcs, Sinit=Sinit,
                                      Smax=Smax)
        Sinit = S0
    return 1 - J


def sim_wiener_group(mc, t, GL, LL, Nmcs=1000, Sinit=0, Smax=0):
    """
    Calculate probability of failure between two maintenance tasks
    (approach by Poetzelberger).
    """
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
        A = np.matmul(np.matmul(M, Dsqrt), u)
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


def gen_group(mc, t, sep):
    """
    Generator for sequences of operating modes between maintenance
    tasks.
    """
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
    """Calculate mue, sd for each task/mode combination."""
    GL = {}
    LL = {}
    for i in stn.I[j]:
        for k in stn.O[j]:
            taskmode = i + "-" + k
            GL[taskmode] = stn.deg[j].get_mu(taskmode)
            LL[taskmode] = (stn.deg[j].get_sd(taskmode))**2
    # TODO: move default values for mue, sd to stn
    GL["None-None"] = 0
    LL["None-None"] = 0.05**2
    GL["M-M"] = 0
    LL["M-M"] = 0.05**2
    return GL, LL


def get_deg_profile(profile, stn, j, dT, dt=1/10, N=1, Sinit=0, S0=0):
    """Get profile of D, sd, and Mt (for naive approach)."""
    Darr = np.zeros((3, 0))
    t = 0
    for taskmode in profile:
        m = 0
        # TODO: move default values for mue, sd to stn
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
            tend = t + stn.p[i, j, k]
            mue, sd = stn.deg[j].get_dist(i + "-" + k, dt)
        tend = int(tend)
        np.array([[1, 2], [3, 4]])
        Darr = np.concatenate((Darr,
                               np.array([
                                         [mue for i
                                          in range(int(t/dt),
                                                   int(tend/dt))],
                                         [sd for i
                                          in range(int(t/dt),
                                                   int(tend/dt))],
                                         [m]+[0 for i
                                              in range(int(t/dt)+1,
                                                       int(tend/dt))]])),
                              axis=1)
        t = tend
    return Darr


def check_feasibility_lambda(lam, N, delta):
    lhs = 1/(N+1)*floor((N+1)/N*((N-1)/lam**2 + 1))
    if lhs <= delta:
        return lam
    else:
        return 10000000
