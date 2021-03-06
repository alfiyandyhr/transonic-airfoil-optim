#Loading variables from config.dat
#Outputting variables as python variables loaded in main.py
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/5/2021
#####################################################################################################
import numpy as np
from BSpline import BSplineFromControlPoints

# -------------------- Parameters Loader ---------------------#
def load_vars():
	"""Loading all variables and storing them in a dictionary"""
	with open('config.dat') as f:
		content = f.readlines()
		config = {}
		for line in content:
			if line.startswith('%'):
				continue
			item = line.rstrip().split(' = ')
			config[item[0]] = item[1]
	return config

config = load_vars()

"""Assigning variables"""
#Design of Experiment
pop_size = eval(config['POPULATION_SIZE'])

#Problem Definition
n_var = eval(config['NUMBER_OF_ACTIVE_CONTROL_POINTS'])
perturbation = eval(config['PERTURBATION'])
degree = eval(config['BSPLINE_DEGREE'])
n_obj = eval(config['N_OBJ'])
n_constr = eval(config['N_CONSTR'])

#Mesh
pw_port = eval(config['PW_PORT'])
mesh_type = config['MESH_TYPE']
con_dimension = eval(config['AIRFOIL_DIM'])
farfield_dim = eval(config['FARFIELD_DIM'])
farfield_radius = eval(config['FARFIELD_DIST'])
step_dim = eval(config['STEP_DIM'])
first_spacing = eval(config['FIRST_SPACING'])
size_field_decay = eval(config['SIZE_FIELD_DECAY'])
le_spacing = eval(config['LE_SPACING'])
te_spacing = eval(config['TE_SPACING'])
algorithm = config['ALGORITHM'].title()
if algorithm == 'Advancing_Front':
	algorithm = 'AdvancingFront'
elif algorithm == 'Advancing_Front_Ortho':
	algorithm = 'AdvancingFrontOrtho'
solver = config['SOLVER']
dimension = eval(config['DIMENSION'])

#Neural Network configuration
N_Epoch = eval(config['N_EPOCH'])
N_Neuron = eval(config['N_NEURON'])
lr = eval(config['LEARNING_RATE'])
train_ratio = eval(config['TRAIN_RATIO'])
batchrate = eval(config['BATCHRATE'])
number_of_updates = eval(config['NO_OF_UPDATES'])

#Optimization configuration on the trained NN model
algorithm_name = config['OPTIMIZATION_ALGORITHM']
initial_sampling_method_name = config['SAMPLING_METHOD']
selection_operator_name = config['SELECTION_OPERATOR']
crossover_operator_name = config['CROSSOVER_OPERATOR']
prob_c = eval(config['CROSSOVER_PROBABILITY'])
eta_c = eval(config['ETA_CROSSOVER'])
mutation_operator_name = config['MUTATION_OPERATOR']
eta_m = eval(config['ETA_MUTATION'])
termination_name = config['TERMINATION']
n_gen = eval(config['NUMBER_OF_GENERATION'])

# -------------------- Baseline Parameters ---------------------#

#Reference area calculation
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

control_upper = np.genfromtxt('Designs/baseline/rae2282_base_control.dat',
							   usecols=(0,1),
							   skip_header=int(n_var/2)+1)
control_lower = np.genfromtxt('Designs/baseline/rae2282_base_control.dat',
							   usecols=(0,1),
							   skip_footer=int(n_var/2)+2)

# ref_control = np.concatenate((control_upper, control_lower))
# ref_area = evaluate_area(ref_control)
ref_area = 0.0812352023725

control_x = np.genfromtxt('Designs/baseline/rae2282_base_control.dat',usecols=0).reshape((n_var+3,1))

# #Design space
# d1 = [0.0,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.01,0.002,0.0]
# d2 = [0.0,0.002,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.01]

# y_upper_max = control_upper[:,1] + np.array(d1) 
# y_lower_max = control_lower[:,1] - np.array(d2)
# y_upper_min = control_upper[:,1] - np.array(d1) 
# y_lower_min = control_lower[:,1] + np.array(d2)

# control_max = np.concatenate((control_lower,control_upper),axis=0)
# control_min = np.concatenate((control_lower,control_upper),axis=0)
# control_max[0:14,1] = y_lower_max
# control_max[14:29,1] = y_upper_max
# control_min[0:14,1] = y_lower_min
# control_min[14:29,1] = y_upper_min

# np.savetxt('Designs/baseline/control_max.dat',control_max)
# np.savetxt('Designs/baseline/control_min.dat',control_min)

# bspline = BSplineFromControlPoints(degree=degree)
# bspline.create('Designs/baseline/control_max.dat')
# bspline.rotate_and_dilate('Designs/baseline/bspline_points_max.dat')
# bspline.create('Designs/baseline/control_min.dat')
# bspline.rotate_and_dilate('Designs/baseline/bspline_points_min.dat')

# bspline_points_max = np.genfromtxt('Designs/baseline/bspline_points_max.dat')
# bspline_points_min = np.genfromtxt('Designs/baseline/bspline_points_min.dat')