#Generating airfoil designs
#Outputting 50 random designs with 30 design variables using Latin Hypercube Sampling
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/24/2021
#####################################################################################################
from LoadVars import *
from AirfoilDesign import *
from Sampling import *
from BSpline import * 
from Mesh import *
import matplotlib.pyplot as plt

from SaveOutput import save
from calc import calc_area, calc_y_diff
from su2 import su2_cfd
from scrap import scrap_initial_training_data

import os
import numpy as np
#####################################################################################################
#Importing airfoil baseline
design = AirfoilDesign(design_number=design_number,control_points=control_points)
design.import_baseline(
	baseline_path='Designs/rae2282_base.dat',
	baseline_control_path='Designs/rae2282_base_control.dat')
####################################################################################################
#Creating lhs matrix
sampling = Sampling(design_variable=control_points,samples=design_number)
matrix = sampling.make_matrix()

#Creating random airfoil control points coordinates
design.create_random(sampling_matrix=matrix, perturbation=perturbation)
design.save_random()

#Creating BSpline from random airfoil control points
bspline = BSplineFromControlPoints(degree=degree, design_number=design_number)
bspline.create()
bspline.to_pointwise_format()
#####################################################################################################
#Matplotlib
plt.plot(design.uiuc_data_upper[:,0], design.uiuc_data_upper[:,1], 'b-', markersize = 3, label='RAE2282 - Baseline')
plt.plot(design.uiuc_data_lower[:,0], design.uiuc_data_lower[:,1], 'b-', markersize = 3)
plt.plot(design.control_upper[:,0], design.control_upper[:,1], 'go', markersize = 3, label='RAE2282 - B Spline Control Points')
plt.plot(design.control_lower[:,0], design.control_lower[:,1], 'go', markersize = 3)
plt.plot(bspline.bspline_points[:,0],bspline.bspline_points[:,1],'r-',markersize = 3, label='B Spline')
plt.plot(design.x_upper, design.y_upper[49], 'ro', markersize = 3, label='Random Control Points')
plt.plot(design.x_lower, design.y_lower[49], 'ro', markersize = 3)
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 0.2])
plt.title('Airfoils')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right")
plt.show()
#####################################################################################################
#Prepare the folders to save mesh files
for i in range(design_number):
	os.makedirs('Meshes/gen_1/random_design' + str(i))

#Prepare the SU2 config files for every mesh folders
with open('Meshes/baseline/inv_transonic_airfoil.cfg', 'r') as f:
	su2_cfg = f.read()

for i in range(design_number):
	with open('Meshes/gen_1/random_design' + str(i) + '/inv_transonic_airfoil.cfg', 'w') as f:
		f.write(su2_cfg.replace('rae2282_base.su2','random_design' + str(i) + '.su2'))
#####################################################################################################
"""
Initialization
"""
parent_pop = np.concatenate((design.y_upper,design.y_lower),axis=1)
parent_pop = np.delete(parent_pop,(0,15,16,31),axis=1)

#Area calculation
ref_control = np.concatenate((design.control_upper, design.control_lower),axis=0)
ref_area = calc_area(ref_control)

airfoil_area = []
x_control = np.concatenate((design.x_upper,design.x_lower),axis=0).reshape((32,1))
for i in range(design_number):
	y_control = np.concatenate((design.y_upper[i],design.y_lower[i]),axis=0).reshape((32,1))
	xy_control = np.concatenate((x_control,y_control),axis=1)
	airfoil_area.append(calc_area(xy_control))
airfoil_area = np.array(airfoil_area).reshape(design_number,1)

parent_pop_eval = -(airfoil_area - ref_area)

array = np.concatenate((design.control_upper[:,1],design.control_lower[:,1]))
y_diff_ref = calc_y_diff(array)
y_diff_ref = np.delete(y_diff_ref, (0,int(control_points/2)+1), axis=0)

y_diffs = np.zeros((design_number,int(control_points/2)+2))
for i in range(design_number):
	y_control = np.concatenate((design.y_upper[i],design.y_lower[i])).reshape((32,1))
	y_diff = calc_y_diff(y_control).reshape(-1)
	y_diffs[i,:] = y_diff
y_diffs = np.delete(y_diffs, (0,int(control_points/2)+1), axis=1)

parent_pop_eval = np.concatenate((parent_pop_eval,y_diffs),axis=1)
parent_pop_eval = np.concatenate((np.zeros((50,3)),parent_pop_eval),axis=1)
#####################################################################################################
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

#Meshing all random designs
for i in range(design_number):
	#If the random design doesn't have intersection between curves
	if all(y_diffs[i]>0):
		airfoil_mesh_random = AirfoilMesh(
			file_in='bspline_points_random' + str(i) + '.dat',
			file_out='random_design' + str(i) + '/random_design' + str(i) +'.su2',
			con_dimension=con_dimension,
			le_spacing=le_spacing,
			te_spacing=te_spacing,
			solver=solver,
			dim=dimension,
			mesh_type=mesh_type)
		airfoil_mesh_random.unstructured(
			algorithm,
			size_field_decay,
			farfield_radius,
			farfield_dim)
	else:
		parent_pop_eval[i,0] = 1.0
		parent_pop_eval[i,1] = 1.0
		parent_pop_eval[i,2] = 1.0

print(parent_pop_eval[:,0])
#####################################################################################################
CFD Simulations
su2_cfd(dir='Grid_Convergence_Study/baseline', paralel_comp=True, core=4)

for i in range(50):
	su2_cfd(dir='Meshes/random_design' + str(i), paralel_comp=True, core=4)
#####################################################################################################
#Training

scrap_initial_training_data()
