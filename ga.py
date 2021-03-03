#Genetic Algorithms using pymoo
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/13/2021
#####################################################################################################
import numpy as np

from pymoo.model.problem import Problem
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.model.sampling import Sampling
from pymoo.model.evaluator import Evaluator
from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_sampling, get_selection
from pymoo.factory import get_crossover, get_mutation, get_termination

from NeuralNet import calculate
#####################################################################################################
#Disable warning
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

class TransonicAirfoilOptimization(Problem):
	"""This is the optimization problem"""
	def __init__(self,
				 n_var,
				 n_obj,
				 n_constr,
				 xu,xl):
		"""Inheritance from Problem class"""
		self.n_var = n_var
		self.n_obj = n_obj
		self.n_constr = n_constr
		self.xl = xl
		self.xu = xu
		
		super().__init__(n_var=self.n_var,
						 n_obj=self.n_obj,
						 n_constr=self.n_constr,
						 xl=self.xl, xu=self.xu)
	
class EvolutionaryAlgorithm():
	"""Instance for the crossover operator"""
	def __init__(self, name):
		"""Name of the crossover operator"""
		self.name = name
	def setup(self, pop_size, sampling,
			 crossover, mutation):
		#"""Returning the python object"""
		if self.name == 'nsga2':	
			algorithm = NSGA2(pop_size=pop_size,
							  # selection=selection,
							  sampling=sampling,
							  crossover=crossover,
							  mutation=mutation)
		else:
			print('Please enter the algorithm name!\n')

		return algorithm

def define_sampling(name):
		"""Returning the python object of the initial sampling method"""	
		return get_sampling(name)

def define_selection(name):
		"""Returning the python object of selection operator"""	
		if name == 'tournament':
			selection = get_selection(name, func_comp='real_tournament')
		return selection

def define_crossover(name, prob, eta):
		"""Returning the python of the crossover operator"""	
		return get_crossover(name, prob=prob, eta=eta)

def define_mutation(name, eta):
		"""Returning the python object of the mutation operator"""	
		return get_mutation(name, eta=eta)

class StoppingCriteria():
	"""Instance for the termination"""
	def __init__(self, name):
		"""Name of the stopping criteria"""
		self.name = name
	def set_termination(self, n_gen):
		"""Returning the python object"""	
		termination = get_termination(self.name, n_gen)
		return termination

def set_individual(X, F, G, CV):
	"""
	This will return an individual class in pymoo
	"""
	return Individual(X=X, F=F, G=G, CV=CV)

def set_population(n_individuals):
	"""
	This will return a population class in pymoo
	"""
	return Population(n_individuals=n_individuals)

def do_survival(problem, merged_pop, n_survive):
	"""This will merge two pops and return the best surviving pop"""
	Survival = RankAndCrowdingSurvival()

	surviving_pop = Survival.do(problem, merged_pop, n_survive)

	surviving_pop_eval = surviving_pop.get('F')
	surviving_pop_G = surviving_pop.get('G')
	surviving_pop_CV = surviving_pop.get('CV')

	if surviving_pop_G[0] is not None:
		surviving_pop_eval = np.concatenate((surviving_pop_eval, surviving_pop_G, surviving_pop_CV), axis=1)
	else:
		surviving_pop_G = np.zeros((len(surviving_pop_eval),1))
		surviving_pop_eval = np.concatenate((surviving_pop_eval, surviving_pop_G, surviving_pop_CV), axis=1)

	return surviving_pop, surviving_pop_eval

def do_optimization(problem, algorithm, termination,
	verbose=False, seed=1, return_least_infeasible=True):
	"""Conduct optimization process and return optimized solutions"""
	optim = minimize(problem, algorithm, termination,
					 verbose=verbose, seed=seed,
					 return_least_infeasible=return_least_infeasible)
	return optim
