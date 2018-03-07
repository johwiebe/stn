#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate random degradation signals
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed


def check_for_task(m, stn, dt, j, ti):
    tend = m.sb.TIME[ti] + m.sb.dT
    for i in stn.I[j]:
        for k in stn.O[j]:
            W = m.sb.W[i, j, k, m.sb.TIME[ti]]()
            if W > 0.5:
                D = stn.D[i, j, k]*dt/stn.p[i, j, k]
                sd = 0.27*D/(np.sqrt(dt/stn.p[i, j, k]))
                tend = m.sb.TIME[ti] + stn.p[i, j, k]
                return [D, sd, tend, 0]
    M = m.sb.M[j, m.sb.TIME[ti]]()
    if M > 0.5:
        tend = m.sb.TIME[ti] + stn.tau[j]
    D = 0
    sd = 0.05*np.sqrt(dt)
    return [D, sd, tend, M]


def get_profile(model, j, dt=1/10, N=1, Sinit=0, S0=0):
    stn = model.stn
    Ns = np.floor(model.sb.T/dt).astype(np.int)
    Darr = np.zeros((3, Ns))
    m_ind = 0
    if (len(model.m_list) > 0):
        m = model.m_list[m_ind]
    else:
        m = model.model
    tend = 0
    for s in range(0, Ns):
        t = s*dt
        if tend <= t:
            if (t > m.sb.T):
                m_ind += 1
                if m_ind < len(model.m_list):
                    m = model.m_list[m_ind]
            ti = np.sum([m.sb.TIME <= t]) - 1
            dat = check_for_task(m, stn, dt, j, ti)
            tend = dat[2]
        Darr[0, s] = dat[0]
        Darr[1, s] = dat[1]
        Darr[2, s] = dat[3]
    return Darr


def simulate_wiener7(Darr, j, dt=1/10, N=1, Ns=1,
                     Sinit=0, S0=0, Rmax=0, plot=False):
    np.random.seed()
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


def simulate_wiener6(Darr, j, dt=1/10, N=1, Ns=1,
                     Sinit=0, S0=0, Rmax=0, plot=False):
    np.random.seed()
    Ninf = 0
    for n in range(0, N):
        S = Sinit
        for s in range(0, Ns):
            if Darr[2, s] < 0.5:
                dS = np.random.normal(loc=Darr[0, s], scale=Darr[1, s])
                S = S + dS
            else:
                S = S0
            if S >= Rmax:
                Ninf += 1
                break
    return Ninf


def simulate_wiener5(Darr, j, dt=1/10, N=1, Ns=1,
                     Sinit=0, S0=0, Rmax=0, plot=False):
    np.random.seed()
    S = np.zeros((N, Ns))
    Slast = Sinit
    Ninf = 0
    for n in range(0, N):
        Slast = Sinit
        for s in range(0, Ns):
            if Darr[2, s] < 0.5:
                dS = np.random.normal(loc=Darr[0, s], scale=Darr[1, s])
                if s > 0:
                    Slast = S[n, s - 1]
                S[n, s] = Slast + dS
            else:
                S[n, s] = S0
        if np.sum([S[n, :] >= Rmax]) > 0:
            Ninf += 1
    return Ninf


def simulate_wiener4(model, j, dt=1/10, N=1, Sinit=0, S0=0, plot=False):
    stn = model.stn
    np.random.seed()
    Ns = np.floor(model.sb.T/dt).astype(np.int)
    S = np.zeros((N, Ns))
    Darr = np.zeros((3, Ns))
    TIME = np.array(range(0, S.shape[1])) * dt
    m_ind = 0
    if (len(model.m_list) > 0):
        m = model.m_list[m_ind]
    else:
        m = model.model
    Slast = Sinit
    tend = 0
    for s in range(0, Ns):
        t = s*dt
        if tend <= t:
            if (t > m.sb.T):
                m_ind += 1
                if m_ind < len(model.m_list):
                    m = model.m_list[m_ind]
            ti = np.sum([m.sb.TIME <= t]) - 1
            dat = check_for_task(m, stn, dt, j, ti)
            tend = dat[2]
        Darr[0, s] = dat[0]
        Darr[1, s] = dat[1]
        Darr[2, s] = dat[3]
    Ninf = 0
    for n in range(0, N):
        Slast = Sinit
        for s in range(0, Ns):
            if Darr[2, s] < 0.5:
                dS = np.random.normal(loc=Darr[0, s], scale=Darr[1, s])
                if s > 0:
                    Slast = S[n, s - 1]
                S[n, s] = Slast + dS
            else:
                S[n, s] = S0
        if np.sum([S[n, :] >= stn.Rmax[j]]) > 0:
            Ninf += 1
        if plot:
            plt.plot(TIME, S[n, :])
    if plot:
        plt.plot(TIME, [stn.Rmax[j] for t in TIME])
        plt.show()
    return Ninf


def simulate_wiener3(model, j, dt=1/10, N=1, Sinit=0, S0=0, plot=False):
    stn = model.stn
    np.random.seed()
    Ns = np.floor(model.sb.T/dt).astype(np.int)
    S = np.zeros((N, Ns))
    Darr = np.zeros((2, Ns))
    TIME = np.array(range(0, S.shape[1])) * dt
    Ninf = 0
    for n in range(0, N):
        m_ind = 0
        if (len(model.m_list) > 0):
            m = model.m_list[m_ind]
        else:
            m = model.model
        Slast = Sinit
        tend = 0
        for s in range(0, S.shape[1]):
            t = s*dt
            if tend <= t:
                if (t > m.sb.T):
                    m_ind += 1
                    if m_ind < len(model.m_list):
                        m = model.m_list[m_ind]
                ti = np.sum([m.sb.TIME <= t]) - 1
                dat = check_for_task(m, stn, dt, j, ti)
                D = dat[0]
                sd = dat[1]
                tend = dat[2]
                M = dat[3]
            if M < 0.5:
                dS = np.random.normal(loc=D, scale=sd)
                if s > 0:
                    Slast = S[n, s - 1]
                S[n, s] = Slast + dS
            else:
                S[n, s] = S0
        if np.sum([S[n, :] >= stn.Rmax[j]]) > 0:
            Ninf += 1
        if plot:
            plt.plot(TIME, S[n, :])
    if plot:
        plt.plot(TIME, [stn.Rmax[j] for t in TIME])
        plt.show()
    return Ninf


def simulate_wiener2(model, j, dt=1/10, N=1, Sinit=0, S0=0):
    stn = model.stn
    np.random.seed()
    # N = 1
    S = np.zeros((N, np.floor(model.sb.T/dt).astype(np.int)))
    TIME = np.array(range(0, S.shape[1])) * dt
    Ninf = 0
    for n in range(0, N):
        m_ind = 0
        if (len(model.m_list) > 0):
            m = model.m_list[m_ind]
        else:
            m = model.model
        Slast = Sinit
        tilast = -1
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
        # plt.plot(TIME, S[n, :])
        if np.sum([S[n, :] >= stn.Rmax[j]]) > 0:
            Ninf += 1
    #     plt.plot(TIME, S[n, :])
    # plt.plot(TIME, [stn.Rmax[j] for t in TIME])
    # plt.show()
    return Ninf


def simulate_wiener(model, j, dt=1/10, N=1, Sinit=0, S0=0):
    Ninfeasible = 0
    np.random.seed()
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


def simulate_deg_old(N=1, *args, **kwargs):
    Ncpus = 7
    Nlist = np.ones((1, Ncpus))*np.floor(N/Ncpus)
    Nlist[0, 0] += N % Ncpus
    st = time.time()
    inflist = Parallel(n_jobs=Ncpus)(delayed(simulate_wiener4)(N=int(Ni),
                                                               *args,
                                                               **kwargs)
                                     for Ni in Nlist[0, :])
    print("Time taken:"+str(time.time()-st))
    Ninf = np.sum(inflist)
    return Ninf/N*100


def simulate_deg(N, model, j, dt=1/10, *args, **kwargs):
    Ncpus = 7
    Nlist = np.ones((1, Ncpus))*np.floor(N/Ncpus)
    Nlist[0, 0] += N % Ncpus
    Ns = np.floor(model.sb.T/dt).astype(np.int)
    st = time.time()
    Darr = get_profile(model, j, dt=dt, **kwargs)
    Rmax = model.stn.Rmax[j]
    inflist = Parallel(n_jobs=Ncpus)(delayed(simulate_wiener7)(Darr, j,
                                                               N=int(Ni),
                                                               Ns=Ns,
                                                               Rmax=Rmax,
                                                               *args,
                                                               **kwargs)
                                     for Ni in Nlist[0, :])
    print("Time taken:"+str(time.time()-st))
    Ninf = np.sum(inflist)
    return Ninf/N*100
