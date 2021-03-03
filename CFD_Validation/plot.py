import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = np.genfromtxt('cp_experiment.dat',
					 skip_header=6)

data_SU2 = pd.read_csv('surface_flow.csv', header=0,
						usecols=['x','Pressure_Coefficient'])

data_SU2 = data_SU2.to_numpy()

idx = np.where(data_SU2==0)
add = data_SU2[idx[0][0]].reshape(1,2)
data_SU2 = np.delete(data_SU2, (idx[0][0]),axis=0)

data_SU2 = np.concatenate((data_SU2,add,data_SU2[0].reshape(1,2)),axis=0)

plt.plot(data[:,0],data[:,1],'o',markersize=3)
plt.plot(data_SU2[:,0],-data_SU2[:,1],'-',markersize=3)
plt.show()
