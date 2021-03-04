import matplotlib.pyplot as plt
import numpy as np

# data = np.genfromtxt('dv_lhs.dat')

control_max = np.genfromtxt('control_max.dat')
control_min = np.genfromtxt('control_min.dat')

# plt.plot(data[:,0],data[:,1],'o')

plt.plot(control_max[:,0],control_max[:,1],'o',markersize=2)
plt.plot(control_min[:,0],control_min[:,1],'o',markersize=2)
plt.show()