#Plotting the results
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/10/2021
#####################################################################################################
from LoadVars import *
import numpy as np
import matplotlib.pyplot as plt

#Plot
if initial_pop_plot or best_pop_plot or all_pop_plot:
	if all_pop_plot:
		all_pop = np.genfromtxt('Output/all_pop_FGCV.dat', delimiter=' ')
		all_pop_feasible = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
		all_pop_infeasible = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
		if len(all_pop_infeasible)>0:
			plt.plot(all_pop_infeasible[:,0],all_pop_infeasible[:,1],'ro', markersize=4, label='Infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(all_pop_feasible[:,0],all_pop_feasible[:,1],'bo', markersize=4, label='Feasible')

	if initial_pop_plot:
		initial_pop = np.genfromtxt('Output/initial_pop_FGCV.dat', delimiter=' ')
		initial_pop_feasible = np.delete(initial_pop, np.where(initial_pop[:,-1]>0.0), axis=0)
		initial_pop_infeasible = np.delete(initial_pop, np.where(initial_pop[:,-1]==0.0), axis=0)
		if len(initial_pop_infeasible)>0:
			plt.plot(initial_pop_infeasible[:,0], initial_pop_infeasible[:,1], 'ro', label='Initial solutions - infeasible')
		if len(initial_pop_feasible)>0:
			plt.plot(initial_pop_feasible[:,0], initial_pop_feasible[:,1], 'bo', label='Initial solutions - feasible')
	
	if best_pop_plot:
		best_pop = np.genfromtxt('Output/final_pop_FGCV.dat', delimiter=' ')
		best_pop_feasible = np.delete(best_pop, np.where(best_pop[:,-1]>0.0), axis=0)
		best_pop_infeasible = np.delete(best_pop, np.where(best_pop[:,-1]==0.0), axis=0)
		if len(best_pop_infeasible)>0:
			plt.plot(best_pop_infeasible[:,0], best_pop_infeasible[:,1], 'rx', label='Best optimal solutions - infeasible')
		if len(best_pop_feasible)>0:	
			plt.plot(best_pop_feasible[:,0], best_pop_feasible[:,1], 'bx', label='Best optimal solutions - feasible')
	
	plt.title(f'Objective functions space')
	plt.xlabel("F1")
	plt.ylabel("F2")
	plt.legend(loc="upper right")
	plt.show()

if hv_plot:

	HV = np.genfromtxt('Output/HV.dat',
		 skip_header=0, skip_footer=0, delimiter=' ')

	plt.plot(HV[:,1],HV[:,0])
	plt.title(f'HV History')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("HV value")
	plt.legend(loc="upper right")
	plt.show()

#####################################################################################################

all_pop = np.genfromtxt('Output/all_pop_FGCV.dat', delimiter=' ')

pop_1 = all_pop[0:100]
pop_2 = all_pop[100:200]
pop_3 = all_pop[200:300]

pop_1_feas = np.delete(pop_1, np.where(pop_1[:,-1]>0.0)[0], axis=0)
pop_1_infeas = np.delete(pop_1, np.where(pop_1[:,-1]==0.0)[0], axis=0)
pop_2_feas = np.delete(pop_2, np.where(pop_2[:,-1]>0.0)[0], axis=0)
pop_2_infeas = np.delete(pop_2, np.where(pop_2[:,-1]==0.0)[0], axis=0)
pop_3_feas = np.delete(pop_3, np.where(pop_3[:,-1]>0.0)[0], axis=0)
pop_3_infeas = np.delete(pop_3, np.where(pop_3[:,-1]==0.0)[0], axis=0)

all_pop_feas = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
all_pop_infeas = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)

plt.plot(pop_1_feas[:,0],pop_1_feas[:,1],'bo')
plt.plot(pop_2_feas[:,0],pop_2_feas[:,1],'ro')
plt.plot(pop_3_feas[:,0],pop_3_feas[:,1],'ko')
plt.plot(pop_1_infeas[:,0],pop_1_infeas[:,1],'bx')
plt.plot(pop_2_infeas[:,0],pop_2_infeas[:,1],'rx')
plt.plot(pop_3_infeas[:,0],pop_3_infeas[:,1],'kx')

# plt.plot(all_pop_feas[:,0],all_pop_feas[:,1],'ro')
# plt.plot(all_pop_infeas[:,0],all_pop_infeas[:,1],'rx')
plt.show()