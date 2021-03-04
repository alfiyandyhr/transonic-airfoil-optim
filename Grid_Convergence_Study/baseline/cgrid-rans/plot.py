import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Cd_vs_GridCells.dat',
					 skip_header=18,
					 skip_footer=0)

drag_counts = data[:,0]/0.0001

CL = data[:,1]/0.01

CM = data[:,2]/0.001

N = data[:,3]

# fig, ax = plt.subplots()
# ax.plot(N,drag_counts,'-')
# ax.scatter(N,drag_counts,c='b')
# ax.annotate('Tiny (126.02)', (N[0]+2000,drag_counts[0]))
# ax.annotate('Extra Coarse (114.34)', (N[1]+3000,drag_counts[1]))
# ax.annotate('Coarse (107.33)', (N[2]+3000,drag_counts[2]))
# ax.annotate('Medium (103.90)', (N[3]+500,drag_counts[3]+0.5))
# ax.annotate('Fine (102.55)', (N[4]+2000,drag_counts[4]+0.5))
# ax.annotate('Extra Fine\n(101.77)', (N[5]-10000,drag_counts[5]-3.4))
# ax.annotate('Super Fine\n(101.41)', (N[6]-11000,drag_counts[6]+1.0))
# ax.set_ylabel('Drag Counts')
# ax.set_xlabel('Number of grid cells')
# plt.xlim()
# plt.ylim([95,128])
# plt.title('GCS - Drag')
# plt.grid(b=True)
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(N,CL,'-')
# ax.scatter(N,CL,c='b')
# ax.annotate('Tiny (87.186)', (N[0]+2000,CL[0]))
# ax.annotate('Extra Coarse (88.620)', (N[1]+2000,CL[1]-0.05))
# ax.annotate('Coarse (88.999)', (N[2]+2000,CL[2]-0.1))
# ax.annotate('Medium (89.227)', (N[3]+2000,CL[3]-0.1))
# ax.annotate('Fine (89.375)', (N[4]+2000,CL[4]-0.1))
# ax.annotate('Extra Fine\n(89.503)', (N[5]-7000,CL[5]-0.25))
# ax.annotate('Super Fine\n(89.532)', (N[6]-11000,CL[6]-0.25))
# ax.set_ylabel('CL x 100')
# ax.set_xlabel('Number of grid cells')
# plt.xlim()
# plt.title('GCS - Lift')
# plt.grid(b=True)
# plt.show()

fig, ax = plt.subplots()
ax.plot(N,CM,'-')
ax.scatter(N,CM,c='b')

ax.annotate('Tiny\n(126.155)', (N[0]+2000,CM[0]-0.01))
ax.annotate('Extra Coarse\n(127.211)', (N[1]+2000,CM[1]-0.1))
ax.annotate('Coarse\n(126.799)', (N[2]+2000,CM[2]))
ax.annotate('Medium\n(126.576)', (N[3]+2000,CM[3]+0.02))
ax.annotate('Fine\n(126.549)', (N[4]+2000,CM[4]-0.09))
ax.annotate('Extra Fine\n(126.589)', (N[5]-12000,CM[5]+0.03))
ax.annotate('Super Fine\n(126.556)', (N[6]-11000,CM[6]+0.03))

ax.set_ylabel('CM x 1000')
ax.set_xlabel('Number of grid cells')
plt.xlim()
# plt.ylim([124.85, 125.9])
plt.title('GCS - Moment')
plt.grid(b=True)
plt.show()