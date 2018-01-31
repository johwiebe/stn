#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate random degradation signals
"""
import numpy as np
import matplotlib.pyplot as plt


def simulate_wiener(m, j, dt=1/10, N=1, Sinit=0, S0=0):
    # S = np.zeros((N, np.floor(m.pb.T/dt).astype(np.int)))
    S = np.zeros((N, np.floor(m.pb.T/dt).astype(np.int)))
    TIME = np.array(range(0, S.shape[1])) * dt
    Ninfeasible = 0
    plt.figure()
    for n in range(0, N):
        Slast = Sinit
        tilast = -1
        for s in range(0, S.shape[1]):
            dS = 0
            t = s*dt
            if t <= m.sb.T:
                ti = np.sum([m.sb.TIME < t]) - 1
                D = 0
                for i in m.stn.I[j]:
                    for k in m.stn.O[j]:
                        W = 0
                        # TODO: is this for loop necessary? is there a faster
                        # way?
                        for tprime in m.sb.TIME[(m.sb.TIME <= m.sb.TIME[ti])
                                                & (m.sb.TIME
                                                   >= m.sb.TIME[ti]
                                                   - m.stn.p[i, j, k]
                                                   + m.sb.dT)]:
                            W += m.model.sb.W[i, j, k, tprime]()
                        if W > 0.5:
                            D = m.stn.D[i, j, k]*dt/m.stn.p[i, j, k]
                            sd = 0.07*D/(np.sqrt(dt/m.stn.p[i, j, k]))
                        else:
                            sd = 0.05*np.sqrt(dt)
                dS = np.random.normal(loc=D, scale=sd)
                # dS = W
                M = m.model.sb.M[j, m.sb.TIME[ti]]()
            else:
                ti = np.sum([m.pb.TIME < t]) - 1
                mue = 0
                Nsum = 0
                sd = 0
                variance = 0
                H = m.pb.dT
                for i in m.stn.I[j]:
                    for k in m.stn.O[j]:
                        N = m.model.pb.N[i, j, k, m.pb.TIME[ti]]()
                        D = m.stn.D[i, j, k]*dt/m.stn.p[i, j, k]
                        if N > 0.5:
                            Nsum += N
                            H -= N*m.stn.p[i, j, k]
                            mue = mue*(Nsum - N)/Nsum + D*N/Nsum
                            variance = (variance*(Nsum - N)/Nsum
                                        + ((0.07*D)**2)*N/Nsum)
                            sd = np.sqrt(variance)
                mue = mue*(1 - H/m.pb.dT)
                sd = (sd*(m.pb.dT - H) + 0.05*np.sqrt(dt)*H)/m.pb.dT
                dS = (np.random.normal(loc=mue, scale=sd))
                M = m.model.pb.M[j, m.pb.TIME[ti]]()
            if M < 0.5 or tilast == ti:
                if s > 0:
                    Slast = S[n, s - 1]
                S[n, s] = Slast + dS
            else:
                S[n, s] = S0
                tilast = ti
        if np.sum([S[n, :] >= m.stn.Rmax[j]]) > 0:
            Ninfeasible += 1
        plt.plot(TIME, S[n, :])
    import ipdb; ipdb.set_trace() # noqa
    plt.plot(TIME, np.ones(TIME.shape[0])*m.stn.Rmax[j])
    plt.show()
    plt.figure()
    plt.hist(S[:, S.shape[1]-1], density=True, bins=25)
    plt.show()
    return Ninfeasible
