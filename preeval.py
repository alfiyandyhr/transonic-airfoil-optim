import numpy as np

data = np.genfromtxt("Initial_Sampling/preeval.in", skip_footer=1, dtype=int)
ne = data[0]
nx = data[1]
nf = data[2]
ng = data[3]

x = np.genfromtxt("Initial_Sampling/preeval.in", skip_header=1)
g = np.zeros(ng)

f = np.array([12345])
g[0] = -(x[nx-1] - x[0])
g[1] = -(x[nx-2] - x[1])
g[2] = -(x[nx-3] - x[2])
g[3] = -(x[int(nx/2)] - x[int(nx/2)-1])
# g[0] = -1
# g[1] = -1
# g[2] = -1
# g[3] = -1

res = np.concatenate((x,f,g))

np.savetxt("Initial_Sampling/preeval.out", res, newline=" ")