## Plot the effect of changing D on objective function
import sys
sys.path.append('../STN')
from STNdegBlocks import STN
import numpy as np
import csv
import matplotlib.pyplot as plt
import STNsetDefaults

# Vector of nominal D's
D = [1,2,3,4,5,8,1,3,5,3,7,8,4,5,10,2,4,4,2,5,9,2,5,6]
# Maximum relative deviation
eps = 0.1

N = 35
n = 0
data = np.zeros((35,9))
with open("STN-benchmark.csv", 'a', newline='') as csvfile:
    writer =csv.writer(csvfile)
    for n in range(0,N): # !!!
    for d1 in D1range:
        for d2 in D2range:
            for d3 in D3range:
                if d3 > d2 and d2 > d1:
                    data[n, 0] = d1
                    data[n, 1] = d2
                    data[n, 2] = d3
                    stn = STNsetDefaults.setDefaults()
                    stn.ijkdata('Reaction_3', 'Reactor_2', 'Slow', 24, d1)
                    stn.ijkdata('Reaction_3', 'Reactor_2', 'Normal', 21, d2)
                    stn.ijkdata('Reaction_3', 'Reactor_2', 'Fast', 12, d3)
                    Ts = 168
                    dTs = 3
                    Tp = 168*24
                    dTp = 168
                    TIMEs = range(0,Ts,dTs)
                    TIMEp = range(Ts,Tp,dTp)
                    stn.build(TIMEs,TIMEp)
                    stn.loadres('../data/500T-sampR2-35/benchmarkSTN.pyomo')
                    prefix = str(d1)+'-'+str(d2)+'-'+str(d3)
                    feasible = stn.reevaluate(prefix=prefix)
                    costStorage = stn.model.CostStorage()
                    costMaintenance = stn.model.CostMaintenance()
                    costWear = stn.model.CostWear()
                    row = [d1, d2, d3, costStorage, costMaintenance,
                           costWear, feasible]
                    writer.writerow(row)
                    n += 1


