from LoadVars import *
# from Mesh import *
from su2 import su2_cfd
from pymoo.model.evaluator import Evaluator

import numpy as np
import pandas as pd

def evaluate_aero(gen, mesh, CFD_eval):
	"""
	This function evaluates the aerodynamic properties
	like Cd, Cl, Cm in every generation
	input:
		gen: the generation of population (int)
	"""

	#Meshing all random designs
	if mesh:
		for i in range(pop_size):
			airfoil_mesh_random = AirfoilMesh(
				file_in=str(gen) + '/bspline_points/bspline_points_random' + str(i) + '.dat',
				file_out=str(gen) + '/random_design' + str(i) + '/random_design' + str(i) +'.su2',
				con_dimension=con_dimension,
				le_spacing=le_spacing,
				te_spacing=te_spacing,
				solver=solver,
				dim=dimension,
				mesh_type=mesh_type)
			if mesh_type == 'OGRID_UNSTRUCTURED':
				airfoil_mesh_random.ogrid_structured(
					growth_factor=1.2,
					initial_stepsize=0.001,
					step=100)
			elif mesh_type == 'CGRID_STRUCTURED':
				airfoil_mesh_random.cgrid_structured(
					farfield_radius,
					step_dim,
					first_spacing)
			elif mesh_type == 'OGRID_UNSTRUCTURED':
				airfoil_mesh_random.ogrid_unstructured(
					algorithm,
					size_field_decay,
					farfield_radius,
					farfield_dim)
				
	#CFD evaluations
	if CFD_eval:	
		for i in range(pop_size):
			su2_cfd(dir='Solutions/gen_' + str(gen) + '/random_design' + str(i), paralel_comp=True, core=4)

def evaluate_area(array):
	"""
	This function calculates the area of airfoil,
	by summing all the discrete triangles
	Input:
		array of coordinates (x and y)
	Output:
		the area (double)
	"""
	area = 0.0
	#Calculate the upper triangles
	for i in range(int((len(array)+1)/2)-1):
		h1 = array[i,1] - array[len(array)-(i+1),1]
		h2 = array[i+1,1] - array[len(array)-(i+1),1]
		area += 0.5 * (array[i+1,0]-array[i,0]) * max(h1,h2)

	#Calculate the lower triangles
	for i in range(int((len(array)+1)/2),len(array)):
		if i == len(array)-1:
			h1 = array[i,1] - array[0,1]
			h2 = array[0,1] - array[0,1]
			area += 0.5 * (array[0,0]-array[i,0]) * min(h1,h2)
		else:
			h1 = array[i,1] - array[len(array)-(i+1),1]
			h2 = array[i+1,1] - array[len(array)-(i+1),1]
			area += 0.5 * (array[i+1,0]-array[i,0]) * min(h1,h2)

	return area

def evaluate_y_diff(array):
	"""
	This function calculates the difference of y coordinates,
	between 3 points near T.E. and a point near L.E.
	in the upper and lower coordinates
	Input:
		array of y coordinates (flattened)
	Output:
		the array of y_diffs
	"""
	y_diffs = np.zeros((4,1))
	y_diffs[0] = array[n_var+1] - array[1]
	y_diffs[1] = array[n_var] - array[2]
	y_diffs[2] = array[n_var-1] - array[3]
	y_diffs[3] = array[int(n_var/2)+2] - array[int(n_var/2)]

	return y_diffs

def evaluate(gen, pop, mesh, CFD_eval):
	"""
	This function will do the true evaluation and return the array
	which contains F, G and CV
	Input:
		gen = generation
		problem = problem object in pymoo
		pop = population object in pymoo
	Output:
		array of F, G and CV
	"""
	#------------------- CFD evaluation (expensive!!!) ------------------#
	if mesh or CFD_eval:
		evaluate_aero(gen, mesh, CFD_eval)

	pop_eval = np.zeros((pop_size,3)) #for objective functions
	for i in range(pop_size):
		data = pd.read_csv('Solutions/gen_1/random_design'+str(i)+'/history.csv',
			header=0, usecols=['       "CD"       ','       "CL"       ','       "CMz"      '])
		interest = np.array([data.iloc[len(data)-1]])
		pop_eval[i] = interest 

	#--------------------- Area evaluation (cheap) ----------------------#
	airfoil_area = []
	for i in range(pop_size):
		random_design = np.genfromtxt('Designs/gen_'+str(gen)+'/control_points/random_design'+str(i)+'.dat',
   					   				   usecols=(0,1))
		airfoil_area.append(evaluate_area(random_design))
	airfoil_area = np.array(airfoil_area).reshape(pop_size,1)

	pop_eval = np.concatenate((pop_eval,-(airfoil_area - 0.8*ref_area)),axis=1)

	#----------------- T.E. and L.E. constraints (cheap) ----------------#
	y_diffs = np.zeros((pop_size,4))
	for i in range(pop_size):
		random_design = np.genfromtxt('Designs/gen_'+str(gen)+'/control_points/random_design'+str(i)+'.dat',
   					   				   usecols=(1))
		y_diff = evaluate_y_diff(random_design).reshape(-1)
		y_diffs[i,:] = -y_diff

	pop_eval = np.concatenate((pop_eval,y_diffs),axis=1)

	# pop.set('F',pop_eval[:,0:2],'G',pop_eval[:,2:8])

	# pop_eval = pop.get('F')
	# pop_G = pop.get('G')
	# pop_CV = pop.get('CV')

	return pop_eval

def evaluate_cheap_G_from_array(X):
	"""
	This function will do the true evaluation and return the array
	which contains cheap G
	Input:
		X = design variables
	Output:
		pop_eval_cheap_G = array of cheap G (area and y_diffs)
	"""
	zero = np.zeros((len(X),1))

	y = np.concatenate((zero, X[:,0:int(n_var/2)], zero, X[:,int(n_var/2):n_var], zero),axis=1)
	x = control_x

	#--------------------- Area evaluation (cheap) ----------------------#
	airfoil_area = []
	for indiv in range(pop_size):
		random_design = np.concatenate((x,y[indiv].reshape((n_var+3,1))),axis=1)
		airfoil_area.append(evaluate_area(random_design))
	airfoil_area = np.array(airfoil_area).reshape(pop_size,1)

	pop_eval_cheap_G = -(airfoil_area - 0.8*ref_area)

	#----------------- T.E. and L.E. constraints (cheap) ----------------#
	y_diffs = np.zeros((pop_size,4))
	for indiv in range(pop_size):
		y_diff = evaluate_y_diff(y[indiv]).reshape(-1)
		y_diffs[indiv,:] = -y_diff

	pop_eval_cheap_G = np.concatenate((pop_eval_cheap_G,y_diffs),axis=1)

	return pop_eval_cheap_G