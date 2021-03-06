#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/6/2021
#####################################################################################################
from LoadVars import *
from BSpline import * 
# from Mesh import *
from ga import *
from pymoo.model.population import pop_from_array_or_individual
from NeuralNet import NeuralNet, train, calculate
from SaveOutput import save
from eval import evaluate, evaluate_cheap_G_from_array
from performance import calc_hv

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

#Perform cuda computation if NVidia GPU card available
# if torch.cuda.is_available():
# 	device = torch.device('cuda')
# else:
# 	device = torch.device('cpu')

#Erase the comment if you want to use CPU
device = torch.device('cpu')

####################################################################################################

# -------------------- INITIALIZATION --------------------- #

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

#Evaluating initial samples (true eval)
parent_pop_eval = evaluate(gen=1, pop=parent_pop, mesh=False, CFD_eval=False)

save('Output/initial_pop_X.dat', parent_pop.get('X'), header='Generation 1: X')
save('Output/initial_pop_FGCV.dat', parent_pop_eval, header='Generation 1: F, G, CV')
save('Output/all_pop_X.dat', parent_pop.get('X'), header='Generation 1: X')
save('Output/all_pop_FGCV.dat', parent_pop_eval, header='Generation 1: F, G, CV')
save('Data/Training/X.dat', parent_pop.get('X'))
save('Data/Training/OUT.dat', parent_pop_eval[:, 0:n_obj+1])

#Initial performance
HV = [0.0]
HV += [calc_hv(parent_pop_eval[:,range(n_obj)], ref=[1.0,1.0])]

###################################################################################################

# -------------------- INITIAL TRAINING --------------------- #

print('Feeding the training data to the neural net...\n\n')

Model = NeuralNet(D_in=n_var,
				  H=N_Neuron, D=N_Neuron,
				  D_out=n_obj+1).to(device)

print('Performing initial training...\n')

# train(problem=problem,
# 	  model=Model,
#       N_Epoch=N_Epoch,
#       lr=lr,
#       train_ratio=train_ratio,
#       batchrate=batchrate,
#       device=device)

print('\nAn initial trained model is obtained!\n')
print('--------------------------------------------------')

TrainedModel_Problem = TrainedModelProblem(problem, device)

###################################################################################################

# -------------------- OPTIMIZATION ROUTINES ---------------------#
initial_sampling = define_sampling(initial_sampling_method_name)
selection = define_selection(selection_operator_name)
crossover = define_crossover(crossover_operator_name, prob=prob_c, eta=eta_c)
mutation = define_mutation(mutation_operator_name, eta=eta_m)
Survival = RankAndCrowdingSurvival()

#EA settings
EA = EvolutionaryAlgorithm(algorithm_name)
algorithm = EA.setup(pop_size=pop_size-1,
					 sampling=initial_sampling,
					 # selection=selection,
					 crossover=crossover,
					 mutation=mutation)

#Stopping criteria
stopping_criteria_def = StoppingCriteria(termination_name)
stopping_criteria = stopping_criteria_def.set_termination(n_gen=n_gen)

#Obtaining optimal solutions on the initial trained model
print(f'Performing optimization on the initial trained model using {algorithm_name.upper()}\n')
# res =  do_optimization(TrainedModel_Problem,
# 					   algorithm, stopping_criteria,
# 					   verbose=True, seed=1,
# 					   return_least_infeasible=False)
print('--------------------------------------------------')
print('\nOptimal solutions on the initial trained model is obtained!\n')
print('--------------------------------------------------')

###################################################################################################

# -------------------- ITERATIVE TRAININGS AND OPTIMIZATIONS --------------------- #

for update in range(number_of_updates):
	#Saving best design variables (X_best) on every trained model
	print(f'Updating the training data to the neural net, update={update+1}\n\n')
	with open('Data/Prediction/X_best.dat','a') as f:
		save(f, res.X, header=f'Generation {update+2}: X') 
	
	with open('Data/Training/X.dat','a') as f:
		save(f, res.X)
	
	#Evaluating X_best (true eval)
	child_pop = pop_from_array_or_individual(res.X)
	child_pop_eval = evaluate(gen=update+1, pop=child_pop, mesh=False, CFD_eval=False)

	with open('Data/Training/OUT.dat', 'a') as f:
		save(f, child_pop_eval[:, 0:n_obj+1]) 

	#Merging parent_pop and child_pop
	merged_pop = Population.merge(parent_pop, child_pop)

	#Survival method depending on the algorithm type
	parent_pop, parent_pop_eval = do_survival(problem, merged_pop, n_survive=pop_size-1)

	with open('Output/all_pop_X.dat', 'a') as f:
		save(f, parent_pop.get('X'), header=f'Generation {update+2}: X')

	with open('Output/all_pop_FGCV.dat', 'a') as f:
		save(f, parent_pop_eval, header=f'Generation {update+2}: F, G, CV')

	#Performance measurement for each iteration
	HV  += [calc_hv(parent_pop.get('F')[:,range(n_obj)], ref=[1.0,1.0])]

	#Training neural nets
	print(f'Performing neural nets training, training={update+2}\n')

	Model = torch.load('Data/Prediction/trained_model.pth').to(device)

	train(problem=problem,
		  model=Model,
		  N_Epoch=N_Epoch,
		  lr=lr,
		  train_ratio=train_ratio,
		  batchrate=batchrate,
		  device=device)

	#Optimal solutions
	print('--------------------------------------------------\n')
	print(f'Performing optimization on the trained model using {algorithm_name.upper()}\n')
	res =  do_optimization(TrainedModel_Problem,
						   algorithm, stopping_criteria,
						   verbose=True, seed=1,
						   return_least_infeasible=True)
	print('--------------------------------------------------\n')
	print('Optimal solutions on the trained model is obtained!\n')
	print('--------------------------------------------------\n\n')

###################################################################################################

# -------------------- FINALIZATION --------------------- #

#Evaluating the last X_best (true eval)
child_pop = pop_from_array_or_individual(res.X)
child_pop_eval = evaluate(gen=number_of_updates+2, pop=child_pop, mesh=False, CFD_eval=False)

#Merging parent_pop and child_pop
merged_pop = Population.merge(parent_pop, child_pop)

#Survival method depending on the algorithm type
parent_pop, parent_pop_eval = do_survival(problem, merged_pop, n_survive=pop_size-1)

save('Output/final_pop_X.dat', parent_pop.get('X'), header=f'Generation {number_of_updates+2}: X')
save('Output/final_pop_FGCV.dat', parent_pop_eval, header=f'Generation {number_of_updates+2}: F, G, CV')
with open('Output/all_pop_X.dat', 'a') as f:
	save(f, parent_pop.get('X'), header=f'Generation {number_of_updates+2}: X')
with open('Output/all_pop_FGCV.dat', 'a') as f:
	save(f, parent_pop_eval, header=f'Generation {number_of_updates+2}: F, G, CV') 

#Performance measurement for the last solutions
HV  += [calc_hv(parent_pop_eval[:,range(n_obj)], ref=[1.0,1.0])]

#True evaluation counters
true_eval = [0]
for update in range(number_of_updates+2):
	true_eval += [(pop_size-1)*(update+1)+1]

true_eval = np.array([true_eval]).T
HV = np.array([HV]).T
HV = np.concatenate((HV, true_eval),axis=1)

save('Output/HV.dat', HV, header='HV History: HV value, true eval counters')

print(f'NN based surrogate optimization is DONE! True eval = {(pop_size-1)*(number_of_updates+2)+1}\n')

###################################################################################################

# -------------------- MISCELLANEOUS --------------------- #

# #Prepare the folders to save mesh files
# for i in range(pop_size):
# 	os.makedirs('Solutions/gen_1/random_design' + str(i))

# #Prepare the SU2 config files for every mesh folders
# with open('Solutions/baseline/inv_transonic_airfoil.cfg', 'r') as f:
# 	su2_cfg = f.read()
# for i in range(pop_size):
# 	with open('Solutions/gen_1/random_design' + str(i) + '/inv_transonic_airfoil.cfg', 'w') as f:
# 		f.write(su2_cfg.replace('rae2282_base.su2','random_design' + str(i) + '.su2'))

#####################################################################################################
# #Matplotlib
# plt.plot(design.uiuc_data_upper[:,0], design.uiuc_data_upper[:,1], 'b-', markersize = 3, label='RAE2282 - Baseline')
# plt.plot(design.uiuc_data_lower[:,0], design.uiuc_data_lower[:,1], 'b-', markersize = 3)
# plt.errorbar(design.control_upper[:,0], design.control_upper[:,1],yerr=d1,fmt='ro',markersize=3,ecolor='black',capsize=3, label='Control points')
# plt.errorbar(design.control_lower[:,0], design.control_lower[:,1],yerr=d2,fmt='ro',markersize=3,ecolor='black',capsize=3)
# plt.plot(design.control_upper[:,0], design.control_upper[:,1], 'bo', markersize = 2)
# plt.plot(design.control_lower[:,0], design.control_lower[:,1], 'bo', markersize = 2)
# plt.plot(bspline.bspline_points[:,0],bspline.bspline_points[:,1],'r-',markersize = 3, label='B Spline')
# plt.plot(design.x_upper, design.y_upper[99], 'ro', markersize = 3, label='Random Control Points')
# plt.plot(design.x_lower, design.y_lower[99], 'ro', markersize = 3)
# plt.plot(bspline_points_max[:,0],bspline_points_max[:,1], 'r--', label='Max')
# plt.plot(bspline_points_min[:,0],bspline_points_min[:,1], 'g--', label='Min')
# plt.xlim([-0.2, 1.2])
# plt.xlim([-0.0025, 0.01])
# plt.xlim([0.8, 1.01])
# plt.ylim([-0.04, 0.06])
# plt.ylim([-0.05, 0.05])
# plt.ylim([-0.2, 0.2])
# plt.title('Leading edge constraint')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(loc="upper right")
# plt.show()
####################################################################################################
# #Meshing the baseline
# airfoil_mesh_baseline = AirfoilMesh(
# 	file_in='rae2282_base.dat',
# 	file_out='rae2282_base.su2',
# 	con_dimension=con_dimension,
# 	le_spacing=le_spacing,
# 	te_spacing=te_spacing,
# 	solver=solver,
# 	dim=dimension,
# 	mesh_type=mesh_type)
# #airfoil_mesh_baseline.structured(growth_factor=1.2, initial_stepsize=0.001, step=100)
# airfoil_mesh_baseline.unstructured(
# 	algorithm,
# 	size_field_decay,
# 	farfield_radius,
# 	farfield_dim)
# airfoil_mesh_baseline.cgrid_structured(
# 	farfield_radius,
# 	step_dim,
# 	first_spacing)

# #Meshing first random design
# airfoil_mesh_random_design2 = AirfoilMesh(
# 	file_in='bspline_points_random49.dat',
# 	file_out='random_design2.su2',
# 	con_dimension=con_dimension,
# 	le_spacing=le_spacing,
# 	te_spacing=le_spacing,
# 	solver=solver,
# 	dim=dimension,
# 	mesh_type=mesh_type)
# airfoil_mesh_random_design2.unstructured(
# 	algorithm,
# 	size_field_decay,
# 	farfield_radius,
# 	farfield_dim)