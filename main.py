#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/1/2021
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

#Design space
d1 = [0.0,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.01,0.002,0.0]
d2 = [0.0,0.002,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.01]

design.y_upper_max = design.control_upper[:,1] + np.array(d1) 
design.y_lower_max = design.control_lower[:,1] - np.array(d2)
design.y_upper_min = design.control_upper[:,1] - np.array(d1) 
design.y_lower_min = design.control_lower[:,1] + np.array(d2)

design.control_max = np.concatenate((design.control_lower,design.control_upper),axis=0)
design.control_min = np.concatenate((design.control_lower,design.control_upper),axis=0)
design.control_max[0:14,1] = design.y_lower_max
design.control_max[14:29,1] = design.y_upper_max
design.control_min[0:14,1] = design.y_lower_min
design.control_min[14:29,1] = design.y_upper_min

save('Designs/control_max.dat',design.control_max)
save('Designs/control_min.dat',design.control_min)

bspline = BSplineFromControlPoints(degree=degree)
bspline.create('Designs/control_max.dat')
bspline.rotate_and_dilate('Designs/bspline_points_max.dat')
bspline.create('Designs/control_min.dat')
bspline.rotate_and_dilate('Designs/bspline_points_min.dat')

bspline_points_max = np.genfromtxt('Designs/bspline_points_max.dat')
bspline_points_min = np.genfromtxt('Designs/bspline_points_min.dat')
####################################################################################################
#Creating lhs matrix
sampling = Sampling(design_variable=control_points,samples=design_number)
matrix = sampling.make_matrix()

#Creating random airfoil control points coordinates
design.create_random(sampling_matrix=matrix, perturbation=perturbation)
design.save_random()

Creating BSpline from random airfoil control points
bspline = BSplineFromControlPoints(degree=degree)
for i in range(design_number):
	bspline.create(ctrlpts_path='Designs/initial_samples/control_points/random_design'+str(i)+'.dat')
	bspline.rotate_and_dilate(output_path='Designs/initial_samples/bspline_points/bspline_points_random'+str(i)+'.dat')
	bspline.to_pointwise_format(file_path='Designs/initial_samples/bspline_points/bspline_points_random'+str(i)+'.dat')
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
parent_pop = np.delete(parent_pop,(0,int(control_points/2)+1,control_points+2),axis=1)

#Area calculation
ref_control = np.concatenate((design.control_upper, design.control_lower),axis=0)
ref_area = calc_area(ref_control)

airfoil_area = []
x_control = np.concatenate((design.x_upper,design.x_lower),axis=0).reshape((control_points+3,1))
for i in range(design_number):
	y_control = np.concatenate((design.y_upper[i],design.y_lower[i]),axis=0).reshape((control_points+3,1))
	xy_control = np.concatenate((x_control,y_control),axis=1)
	airfoil_area.append(calc_area(xy_control))
airfoil_area = np.array(airfoil_area).reshape(design_number,1)

parent_pop_eval = -(0.8*airfoil_area - ref_area)

#Leading and trailing edge constraints
array = np.concatenate((design.control_upper[:,1],design.control_lower[:,1]))
y_diff_ref = calc_y_diff(array)
y_diff_ref = np.delete(y_diff_ref, (0,int(control_points/2)+1), axis=0)

y_diffs = np.zeros((design_number,int(control_points/2)+2))
for i in range(design_number):
	y_control = np.concatenate((design.y_upper[i],design.y_lower[i])).reshape((control_points+3,1))
	y_diff = calc_y_diff(y_control).reshape(-1)
	y_diffs[i,:] = y_diff
y_diffs = y_diffs[:,[1,int(control_points/2),int(control_points/2)-1,int(control_points/2)-2]]

parent_pop_eval = np.concatenate((parent_pop_eval,y_diffs),axis=1)
parent_pop_eval = np.concatenate((np.zeros((design_number,3)),parent_pop_eval),axis=1)

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
# CFD Simulations
su2_cfd(dir='Grid_Convergence_Study/baseline', paralel_comp=True, core=4)

for i in range(50):
	su2_cfd(dir='Meshes/random_design' + str(i), paralel_comp=True, core=4)
#####################################################################################################
#Training

scrap_initial_training_data()