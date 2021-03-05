import matplotlib.pyplot as plt
import numpy as np

x = np.genfromtxt('dv_lhs.dat')

g = np.zeros((len(x),4))
nx = len(x[0])

g[:,0] = -(x[:,nx-1] - x[:,0])
g[:,1] = -(x[:,nx-2] - x[:,1])
g[:,2] = -(x[:,nx-3] - x[:,2])
g[:,3] = -(x[:,int(nx/2)] - x[:,int(nx/2)-1])


# control_max = np.genfromtxt('control_max.dat')
# control_min = np.genfromtxt('control_min.dat')

# plt.plot(data[:,0],data[:,1],'o')

# plt.plot(control_max[:,0],control_max[:,1],'o',markersize=2)
# plt.plot(control_min[:,0],control_min[:,1],'o',markersize=2)
# plt.show()