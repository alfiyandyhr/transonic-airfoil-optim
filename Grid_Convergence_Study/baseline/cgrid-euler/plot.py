import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Cd_vs_GridCells.dat',
					 skip_header=9,
					 skip_footer=0)

drag_counts = data[:,0]/0.0001

CL = data[:,1]/0.01

CM = data[:,2]/0.001

N = data[:,3]

##############################################################
# fig, ax = plt.subplots()
# ax.plot(N,drag_counts,'-')
# ax.scatter(N,drag_counts,c='b')

# ax.annotate('Extra\nCoarse (157.05)', (N[0]+3000,drag_counts[0]-2))
# ax.annotate('Coarse\n(120.02)', (N[1]+3000,drag_counts[1]))
# ax.annotate('Medium\n(110.40)', (N[2]+500,drag_counts[2]+1.5))
# ax.annotate('Fine\n(110.09)', (N[3]+2000,drag_counts[3]+1))
# ax.annotate('Extra Fine\n(109.65)', (N[4]-10000,drag_counts[4]+1.5))

# ax.set_ylabel('CD (d.c.)')
# ax.set_xlabel('Number of grid cells')
# plt.xlim()
# plt.ylim([105,160])
# plt.title('GCS - Drag')
# plt.grid(b=True)
# plt.show()
##############################################################
# fig, ax = plt.subplots()
# ax.plot(N,CL,'-')
# ax.scatter(N,CL,c='b')

# ax.annotate('Extra Coarse\n(79.604)', (N[0]+2000,CL[0]))
# ax.annotate('Coarse\n(87.517)', (N[1]+2000,CL[1]-1))
# ax.annotate('Medium\n(90.162)', (N[2]+2000,CL[2]-1.2))
# ax.annotate('Fine\n(91.461)', (N[3]+2000,CL[3]-1.2))
# ax.annotate('Extra Fine\n(91.459)', (N[4]-12000,CL[4]-1.2))

# ax.set_ylabel('CL (l.c.)')
# ax.set_xlabel('Number of grid cells')
# plt.xlim()
# plt.title('GCS - Lift')
# plt.grid(b=True)
# plt.show()
##############################################################
fig, ax = plt.subplots()
ax.plot(N,CM,'-')
ax.scatter(N,CM,c='b')

ax.annotate('Extra Coarse\n(114.028)', (N[0]+2000,CM[0]))
ax.annotate('Coarse\n(124.986)', (N[1]+2000,CM[1]-1))
ax.annotate('Medium\n(129.051)', (N[2]+2000,CM[2]-1.3))
ax.annotate('Fine\n(131.593)', (N[3]+2000,CM[3]-1.7))
ax.annotate('Extra Fine\n(131.466)', (N[4]-12000,CM[4]-1.9))

ax.set_ylabel('CM x 1000')
ax.set_xlabel('Number of grid cells')
plt.title('GCS - Moment')
plt.grid(b=True)
plt.show()