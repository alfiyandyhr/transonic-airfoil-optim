#Sampling
#Coded by Alfiyandy Hariansyah
#Tohoku University
#1/25/2021
######################################################################################################
from pyDOE import lhs
######################################################################################################
#criterion arguments: center 'c', maximin 'm', centermaximin 'cm', correlation 'corr'

class Sampling:
	"""Create sampling based on various methods"""
	def __init__(
		self,
		design_variable=28,
		samples=50,
		criterion=None,
		iterations=None):

		self.design_variable = design_variable
		self.samples = samples
		self.criterion = criterion
		self.iterations = iterations

	def make_matrix(self):
	#Create sampling based on Latin Hypercube Sampling Method
		matrix = lhs(self.design_variable,samples=self.samples,criterion=self.criterion,iterations=self.iterations)
		return matrix