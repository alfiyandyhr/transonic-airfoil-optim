#Loading variables from config.dat
#Outputting variables as python variables loaded in main.py
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/4/2021
#####################################################################################################

def load_vars():
	"""Loading all varibales and storing them in a dictionary"""
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
design_number = eval(config['NUMBER_OF_INITIAL_DESIGNS'])
control_points = eval(config['NUMBER_OF_ACTIVE_CONTROL_POINTS'])
perturbation = eval(config['PERTURBATION'])
degree = eval(config['BSPLINE_DEGREE'])
con_dimension = eval(config['AIRFOIL_DIM'])
farfield_dim = eval(config['FARFIELD_DIM'])
farfield_radius = eval(config['FARFIELD_DIST'])
step_dim = eval(config['STEP_DIM'])
first_spacing = eval(config['FIRST_SPACING'])
size_field_decay = eval(config['SIZE_FIELD_DECAY'])
le_spacing = eval(config['LE_SPACING'])
te_spacing = eval(config['TE_SPACING'])
mesh_type = config['MESH_TYPE']
algorithm = config['ALGORITHM'].title()
if algorithm == 'Advancing_Front':
	algorithm = 'AdvancingFront'
elif algorithm == 'Advancing_Front_Ortho':
	algorithm = 'AdvancingFrontOrtho'
solver = config['SOLVER']
dimension = eval(config['DIMENSION'])