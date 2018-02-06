#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate random degradation signals
"""
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def simulate_wiener2(model, j, dt=1/10, N=1, Sinit=0, S0=0):
    stn = model.stn
    S = np.zeros((N, np.floor(model.sb.T/dt).astype(np.int)))
    n = 0
    tilast = -1
    Slast = Sinit
    m_ind = 0
    if (len(model.m_list) > 0):
        m = model.m_list[m_ind]
    else:
        m = model.model
    for s in range(0, S.shape[1]):
        dS = 0
        t = s*dt
        if (t > m.sb.T):
            m_ind += 1
            if m_ind < len(model.m_list):
                m = model.m_list[m_ind]
        ti = np.sum([m.sb.TIME < t]) - 1
        D = 0
        sd = 0.05*np.sqrt(dt)
        for i in stn.I[j]:
            for k in stn.O[j]:
                W = 0
                # TODO: is this for loop necessary?
                # is there a faster way?
                for tprime in m.sb.TIME[(m.sb.TIME
                                         <= m.sb.TIME[ti])
                                        & (m.sb.TIME
                                           >= m.sb.TIME[ti]
                                           - stn.p[i, j, k]
                                           + m.sb.dT)]:
                    W += m.sb.W[i, j, k, tprime]()
                if W > 0.5:
                    D = stn.D[i, j, k]*dt/stn.p[i, j, k]
                    sd = 0.27*D/(np.sqrt(dt/stn.p[i, j, k]))
        dS = np.random.normal(loc=D, scale=sd)
        # dS = W
        M = m.sb.M[j, m.sb.TIME[ti]]()
        if M < 0.5 or tilast == ti:
            if s > 0:
                Slast = S[n, s - 1]
            S[n, s] = Slast + dS
        else:
            S[n, s] = S0
            tilast = ti
    if np.sum([S[n, :] >= stn.Rmax[j]]) > 0:
        return 1
    else:
        return 0


def simulate_wiener(model, j, dt=1/10, N=1, Sinit=0, S0=0):
    Ninfeasible = 0
    stn = model.stn
    S = np.zeros((N, np.floor(model.sb.T/dt).astype(np.int)))
    TIME = np.array(range(0, S.shape[1])) * dt
    plt.figure()
    for n in range(0, N):
        tilast = -1
        Slast = Sinit
        m_ind = 0
        if (len(model.m_list) > 0):
            m = model.m_list[m_ind]
        else:
            m = model.model
        for s in range(0, S.shape[1]):
            dS = 0
            t = s*dt
            if (t > m.sb.T):
                m_ind += 1
                if m_ind < len(model.m_list):
                    m = model.m_list[m_ind]
            ti = np.sum([m.sb.TIME < t]) - 1
            D = 0
            sd = 0.05*np.sqrt(dt)
            for i in stn.I[j]:
                for k in stn.O[j]:
                    W = 0
                    # TODO: is this for loop necessary?
                    # is there a faster way?
                    for tprime in m.sb.TIME[(m.sb.TIME
                                             <= m.sb.TIME[ti])
                                            & (m.sb.TIME
                                               >= m.sb.TIME[ti]
                                               - stn.p[i, j, k]
                                               + m.sb.dT)]:
                        W += m.sb.W[i, j, k, tprime]()
                    if W > 0.5:
                        D = stn.D[i, j, k]*dt/stn.p[i, j, k]
                        sd = 0.27*D/(np.sqrt(dt/stn.p[i, j, k]))
            dS = np.random.normal(loc=D, scale=sd)
            # dS = W
            M = m.sb.M[j, m.sb.TIME[ti]]()
            if M < 0.5 or tilast == ti:
                if s > 0:
                    Slast = S[n, s - 1]
                S[n, s] = Slast + dS
            else:
                S[n, s] = S0
                tilast = ti
        if np.sum([S[n, :] >= stn.Rmax[j]]) > 0:
            Ninfeasible += 1
        plt.plot(TIME, S[n, :])
        if n % 1000 == 0:
            print(str(n)+"/"+str(N))
    plt.plot(TIME, [stn.Rmax[j] for t in TIME])
    plt.title("P = " + str(Ninfeasible/N))
    plt.show()
    return Ninfeasible


def simulate_deg(N=1, *args, **kwargs):
    Nlist = Parallel(n_jobs=7)(delayed(simulate_wiener2)(*args, **kwargs)
                               for i in range(N))
    Ninfeasible = np.sum(Nlist)
    print(str(Ninfeasible)+"/"+str(N))
    return Ninfeasible
