import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Cd_vs_GridCells.dat', skip_header=18)

x = data[:,3]
y1 = data[:,1]
y2 = data[:,0]

fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(x,y1,'r',label='Cl')
ax2.plot(x,y2,'b',label='Cd')
ax.set_xlabel('Number of Grid Cells')
ax.set_ylabel('Cl')
ax2.set_ylabel('Cd')
ax.legend(loc='center right', bbox_to_anchor=(0.7,0.6))
ax2.legend(loc='center right', bbox_to_anchor=(0.7,0.5))
plt.title('Grid Convergence Study of Airfoil Random 2')
plt.show()