## Plot the effect of changing D on objective function
import sys
sys.path.append('../STN')
from STNdegBlocks import STN
import numpy as np
import csv
import matplotlib.pyplot as plt
import STNsetDefaults
import numpy as np

D1range = [1, 2, 3, 4, 5]
D2range = [3, 5, 7, 9, 11]
#D3range = [4, 6, 8, 10, 12, 14]
D3range = [5, 7, 9, 11]

N = 35
n = 0
data = np.zeros((35,14))
for d1 in D1range:
    for d2 in D2range:
        for d3 in D3range:
            if d3 > d2 and d2 > d1:
                data[n, 0] = d1
                data[n, 1] = d2
                data[n, 2] = d3
                n += 1

with open('../data/500T-sampR2-35/STN-eval.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        data[i,3:6] = row[0:3]
        data[i,6] = sum(map(float,row[0:3]))
        data[i,7] = np.mean(data[i,0:3])
with open('../data/500T-sampR2-35/STN-benchmark.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        data[i,8:11] = row[3:6]
        data[i,11] = sum(map(float,row[3:6]))
        data[i,12] = np.sqrt(np.mean((np.array(row[0:3],float) -
                                      np.array([2,5,9],float)) ** 2))
        data[i,13] = row[6] == 'True'

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(data[:,0], data[:,6], 'o')
ax1.set_xlabel("d1 [hr]")
ax1.set_ylabel("Total Cost [$]")
fig1.savefig('cost-vs-d1.png')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(data[:,7], data[:,6], 'o')
ax2.set_xlabel("avg d [hr]")
ax2.set_ylabel("Total Cost [$]")
fig2.savefig('cost-vs-avgd.png')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(data[:,12], data[:,6]-data[:,11], 'o')
ax3.plot(data[data[:,13] == 0,12], data[data[:,13] == 0,6]-data[data[:,13] ==
                                                               0,11], 'ro')
ax3.set_xlabel("MSD d [hr]")
ax3.set_ylabel("Cost diff [$]")
fig3.savefig('costdif-vs-avgd.png')
import ipdb; ipdb.set_trace()
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(data[:,0], data[:,6]-data[:,11], 'o')
ax4.plot(data[data[:,13] == 0,0], data[data[:,13] == 0,6]-data[data[:,13] ==
                                                               0,11], 'ro')
ax4.plot(data[15,0], data[15,6]-data[15,11], 'go')
ax4.set_xlabel("d1 [hr]")
ax4.set_ylabel("Cost diff [$]")
fig4.savefig('costdif-vs-d1.png')
