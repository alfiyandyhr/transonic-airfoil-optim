import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Cd_vs_GridCells.dat',
					 skip_header=38,
					 skip_footer=29)

drag_counts = data[:,0]/0.0001

CL_target = data[:,1]

N = data[:,4]

fig, ax = plt.subplots()
ax.plot(N,drag_counts,'-')
ax.scatter(N,drag_counts,c='b')
ax.annotate('Coarse (107.41)', (N[0]+20000,drag_counts[0]))
ax.annotate('Medium (102.52)', (N[1],drag_counts[1]+0.3))
ax.annotate('Fine (101.52)', (N[2]-90000,drag_counts[2]-0.5))
ax.set_ylabel('Drag counts')
ax.set_xlabel('Number of grid Cells')
plt.xlim()
plt.ylim([100,108])
plt.title('Drag counts at target CL = 0.89311')
plt.grid(b=True)
plt.show()