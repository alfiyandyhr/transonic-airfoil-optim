
#Sampling
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/5/2021
######################################################################################################
from random import shuffle, uniform
import numpy as np
import matplotlib.pyplot as plt
# from Sampling import *

def lhs(design_vars, samples, max_val, min_val):
	"""This will return a matrix of samples using the Latin Hypercube Sampling"""
	matrix = np.zeros((design_vars, samples))

	delta = float((max_val - min_val)/samples) 

	domains = [x for x in range(samples)]
	shuffle(domains)
	domains = [domains]
	for design_var in range(design_vars-1):
		domain = [x for x in range(samples)]
		shuffle(domain)
		domain = [domain]
		domains = np.append(domains, domain, axis=0)
	
	for design_var in range(design_vars):
		for sampling in range(samples):
			matrix[design_var, sampling] = domains[design_var, sampling]*delta + uniform(0,1)*delta

	return matrix.transpose()

#This part compares my code with pyDOE
# my_matrix = lhs(design_vars=2, samples=20, max_val=1, min_val=0)

# sampling = Sampling(design_variable=2, samples=20)
# matrix_pyDOE = sampling.make_matrix()

# plt.plot(my_matrix[:,0],my_matrix[:,1],'ob')
# plt.plot(matrix_pyDOE[:,0],matrix_pyDOE[:,1],'or')
# plt.show()
