import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Cd_vs_GridCells.dat',
					 skip_header=2,
					 skip_footer=2)

drag_counts = data[:,0]/0.0001

CL = data[:,1]/0.01

CM = data[:,2]/0.001

N = data[:,4]

##############################################################
fig, ax = plt.subplots()
ax.plot(N,drag_counts,'-')
ax.scatter(N,drag_counts,c='b')

ax.annotate('Extra Coarse (114.89)', (N[0]+3000,drag_counts[0]))
ax.annotate('Coarse (101.93)', (N[1]+3000,drag_counts[1]))
ax.annotate('Medium (97.72)', (N[2]+500,drag_counts[2]+0.5))
ax.annotate('Fine (96.52)', (N[3]+2000,drag_counts[3]+0.5))
ax.annotate('Extra Fine (95.93)', (N[4]-20000,drag_counts[4]+0.7))

ax.set_ylabel('Drag Counts')
ax.set_xlabel('Number of grid cells')
plt.xlim()
# plt.ylim([94,117])
plt.title('GCS - Drag')
plt.grid(b=True)
plt.show()
##############################################################
fig, ax = plt.subplots()
ax.plot(N,CL,'-')
ax.scatter(N,CL,c='b')

ax.annotate('Extra Coarse (87.631)', (N[0]+2000,CL[0]))
ax.annotate('Coarse (88.556)', (N[1]+2000,CL[1]-0.03))
ax.annotate('Medium (88.858)', (N[2]+2000,CL[2]-0.07))
ax.annotate('Fine (88.966)', (N[3]+2000,CL[3]-0.05))
ax.annotate('Extra Fine (89.027)', (N[4]-22000,CL[4]-0.1))

ax.set_ylabel('CL x 100')
ax.set_xlabel('Number of grid cells')
plt.xlim()
plt.title('GCS - Lift')
plt.grid(b=True)
plt.show()
##############################################################
fig, ax = plt.subplots()
ax.plot(N,CM,'-')
ax.scatter(N,CM,c='b')

ax.annotate('Extra Coarse\n(125.836)', (N[0]+2000,CM[0]-0.06))
ax.annotate('Coarse\n(125.204)', (N[1]+2000,CM[1]+0.01))
ax.annotate('Medium\n(124.959)', (N[2]+2000,CM[2]+0.02))
ax.annotate('Fine\n(124.913)', (N[3]+2000,CM[3]+0.02))
ax.annotate('Extra Fine\n(124.909)', (N[4]-12000,CM[4]+0.03))

ax.set_ylabel('CM x 1000')
ax.set_xlabel('Number of grid cells')
plt.xlim()
# plt.ylim([124.85, 125.9])
plt.title('GCS - Moment')
plt.grid(b=True)
plt.show()