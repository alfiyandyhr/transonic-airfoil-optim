import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Cd_vs_GridCells.dat',
					 skip_header=2,
					 skip_footer=1)

drag_counts = data[:,0]/0.0001

CL = data[:,1]/0.01

N = data[:,2]

# print(drag_counts)

# fig, ax = plt.subplots()
# ax.plot(N,drag_counts,'-')
# ax.scatter(N,drag_counts,c='b')

# ax.annotate('Extra Coarse (114.89)', (N[0]+3000,drag_counts[0]))
# ax.annotate('Coarse (101.93)', (N[1]+3000,drag_counts[1]))
# ax.annotate('Medium (97.72)', (N[2]+500,drag_counts[2]+0.5))
# ax.annotate('Fine (96.52)', (N[3]+2000,drag_counts[3]+0.5))
# ax.annotate('Extra Fine (95.93)', (N[4]-20000,drag_counts[4]+0.7))

# ax.set_ylabel('Drag Counts')
# ax.set_xlabel('Number of grid cells')
# plt.xlim()
# plt.ylim([94,117])
# plt.title('GCS - Drag')
# plt.grid(b=True)
# plt.show()



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