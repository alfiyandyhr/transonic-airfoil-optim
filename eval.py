from LoadVars import *
# from Mesh import *
from su2 import su2_cfd
from pymoo.model.evaluator import Evaluator

import numpy as np

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
			airfoil_mesh_random.cgrid_structured(
				farfield_radius,
				step_dim,
				first_spacing)

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
		problem = problem object in pymoo
		pop = population object in pymoo
	Output:
		array of F, G and CV
	"""
	#------------------- CFD evaluation (expensive!!!) ------------------#
	if mesh or CFD_eval:
		evaluate_aero(gen, mesh, CFD_eval)

	#--------------------- Area evaluation (cheap) ----------------------#
	airfoil_area = []
	for i in range(pop_size):
		random_design = np.genfromtxt('Designs/gen_'+str(gen)+'/control_points/random_design'+str(i)+'.dat',
   					   				   usecols=(0,1))
		airfoil_area.append(evaluate_area(random_design))
	airfoil_area = np.array(airfoil_area).reshape(pop_size,1)

	pop_eval = -(airfoil_area - 0.8*ref_area)

	#----------------- T.E. and L.E. constraints (cheap) ----------------#
	y_diffs = np.zeros((pop_size,4))
	for i in range(pop_size):
		random_design = np.genfromtxt('Designs/gen_'+str(gen)+'/control_points/random_design'+str(i)+'.dat',
   					   				   usecols=(1))
		y_diff = evaluate_y_diff(random_design).reshape(-1)
		y_diffs[i,:] = -y_diff

	pop_eval = np.concatenate((pop_eval,y_diffs),axis=1)


	pop.set('G',pop_eval)
	# pop_eval = pop.get('F')
	# pop_G = pop.get('G')
	# pop_CV = pop.get('CV')


	# if pop_G[0] is not None:
	# 	pop_eval = np.concatenate((pop_eval, pop_G, pop_CV), axis=1)
	# else:
	# 	pop_G = np.zeros((len(pop_eval),1))
	# 	pop_eval = np.concatenate((pop_eval, pop_G, pop_CV), axis=1)

	# return pop_eval
	return pop_eval