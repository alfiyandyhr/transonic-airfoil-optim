#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/5/2021
#####################################################################################################
from LoadVars import *
from BSpline import * 
from Mesh import *
# from ga import *
from pymoo.model.population import pop_from_array_or_individual
from SaveOutput import save
from eval import evaluate

import os
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################
# -------------------- INITIALIZATION ---------------------#

#Initial sampling using LHS with constraint handling
# os.system("gcc -o lhs lhs.c")
# os.system("lhs")

parent_pop = np.genfromtxt('Initial_Sampling/dv_lhs.dat')
y = np.copy(parent_pop)
zero = np.zeros((len(y),1))

y = np.concatenate((zero, y[:,0:int(n_var/2)], zero, y[:,int(n_var/2):n_var], zero),axis=1)
x = control_x
z = np.zeros((len(x),1))

for indiv in range(pop_size):
	save('Designs/gen_1/control_points/random_design' + str(indiv) + '.dat',
		 np.concatenate((x,y[indiv].reshape((n_var+3,1)),z),axis=1))

#Creating BSpline from random airfoil control points
bspline = BSplineFromControlPoints(degree=degree)
for i in range(pop_size):
	bspline.create(ctrlpts_path='Designs/gen_1/control_points/random_design'+str(i)+'.dat')
	bspline.rotate_and_dilate(output_path='Designs/gen_1/bspline_points/bspline_points_random'+str(i)+'.dat')
	bspline.to_pointwise_format(file_path='Designs/gen_1/bspline_points/bspline_points_random'+str(i)+'.dat')

xl = np.genfromtxt('lhs_constr_config.in',
					skip_header=5,
					usecols=0)
xu = np.genfromtxt('lhs_constr_config.in',
					skip_header=5,
					usecols=1)

#Problem definition
problem = TransonicAirfoilOptimization(n_var,
									   n_obj,
									   n_constr,
									   xu, xl)

parent_pop = pop_from_array_or_individual(parent_pop)

parent_pop_eval = evaluate(gen=1, pop=parent_pop, mesh=False, CFD_eval=True)

print(parent_pop.get('G'))

###################################################################################################
# #-------------------- INITIAL TRUE EVALUATION ---------------------#

# #Prepare the folders to save mesh files
# for i in range(pop_size):
# 	os.makedirs('Solutions/gen_1/random_design' + str(i))

# #Prepare the SU2 config files for every mesh folders
# with open('Solutions/baseline/inv_transonic_airfoil.cfg', 'r') as f:
# 	su2_cfg = f.read()

# for i in range(pop_size):
# 	with open('Solutions/gen_1/random_design' + str(i) + '/inv_transonic_airfoil.cfg', 'w') as f:
# 		f.write(su2_cfg.replace('rae2282_base.su2','random_design' + str(i) + '.su2'))

# evaluate()

#####################################################################################################
#Matplotlib
plt.plot(design.uiuc_data_upper[:,0], design.uiuc_data_upper[:,1], 'b-', markersize = 3, label='RAE2282 - Baseline')
plt.plot(design.uiuc_data_lower[:,0], design.uiuc_data_lower[:,1], 'b-', markersize = 3)
plt.errorbar(design.control_upper[:,0], design.control_upper[:,1],yerr=d1,fmt='ro',markersize=3,ecolor='black',capsize=3, label='Control points')
plt.errorbar(design.control_lower[:,0], design.control_lower[:,1],yerr=d2,fmt='ro',markersize=3,ecolor='black',capsize=3)
plt.plot(design.control_upper[:,0], design.control_upper[:,1], 'bo', markersize = 2)
plt.plot(design.control_lower[:,0], design.control_lower[:,1], 'bo', markersize = 2)
plt.plot(bspline.bspline_points[:,0],bspline.bspline_points[:,1],'r-',markersize = 3, label='B Spline')
plt.plot(design.x_upper, design.y_upper[99], 'ro', markersize = 3, label='Random Control Points')
plt.plot(design.x_lower, design.y_lower[99], 'ro', markersize = 3)
plt.plot(bspline_points_max[:,0],bspline_points_max[:,1], 'r--', label='Max')
plt.plot(bspline_points_min[:,0],bspline_points_min[:,1], 'g--', label='Min')
plt.xlim([-0.2, 1.2])
plt.xlim([-0.0025, 0.01])
plt.xlim([0.8, 1.01])
plt.ylim([-0.04, 0.06])
plt.ylim([-0.05, 0.05])
plt.ylim([-0.2, 0.2])
plt.title('Leading edge constraint')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right")
plt.show()

####################################################################################################
#Meshing the baseline
airfoil_mesh_baseline = AirfoilMesh(
	file_in='rae2282_base.dat',
	file_out='rae2282_base.su2',
	con_dimension=con_dimension,
	le_spacing=le_spacing,
	te_spacing=te_spacing,
	solver=solver,
	dim=dimension,
	mesh_type=mesh_type)
#airfoil_mesh_baseline.structured(growth_factor=1.2, initial_stepsize=0.001, step=100)
airfoil_mesh_baseline.unstructured(
	algorithm,
	size_field_decay,
	farfield_radius,
	farfield_dim)
airfoil_mesh_baseline.cgrid_structured(
	farfield_radius,
	step_dim,
	first_spacing)

#Meshing first random design
airfoil_mesh_random_design2 = AirfoilMesh(
	file_in='bspline_points_random49.dat',
	file_out='random_design2.su2',
	con_dimension=con_dimension,
	le_spacing=le_spacing,
	te_spacing=le_spacing,
	solver=solver,
	dim=dimension,
	mesh_type=mesh_type)
airfoil_mesh_random_design2.unstructured(
	algorithm,
	size_field_decay,
	farfield_radius,
	farfield_dim)

print(parent_pop_eval[:,0])
#####################################################################################################
# CFD Simulations
su2_cfd(dir='Grid_Convergence_Study/baseline', paralel_comp=True, core=4)


#####################################################################################################
#Training
