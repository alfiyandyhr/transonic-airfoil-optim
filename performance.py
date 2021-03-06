#Performance Indicator Measurement
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/12/2021
#####################################################################################################
import numpy as np

def calc_hv(pop, ref):
	"""
	This will calculate Hypervolumes from the input array

	Input: The population at a given generation(numpy)
	
	Output: HV value at a given generation(double)

	"""
	#Sorting the population properly
	pop = pop[pop[:,0].argsort()]

	volume = 0.0

	for indiv in range(len(pop)):
		if indiv == len(pop)-1:
			volume += (ref[0] - pop[indiv,0]) * (ref[1] - pop[indiv,1])
			break
		else:
			volume += (ref[1] - pop[indiv,1]) * (pop[indiv+1,0] - pop[indiv,0])

	if volume < 0.0:
		volume = 0.0
		
	return volume

def calc_igd(pop, pf):
	"""
	This will calculate the inverted generational distance

	Input:
		pop: The population at a given generation(numpy)
		pf: Pareto front of the problem
	
	Output: IGD value at a given generation(double)

	"""	
	sum_igd = 0.0
	for indiv_pf in range(len(pf)):
		min_igd = 1.0E5
		for indiv in range(len(pop)):
			igd = np.sqrt(np.power((pf[indiv_pf,0]-pop[indiv,0]),2)+np.power((pf[indiv_pf,0]-pop[indiv,1]),2)) 
			if igd < min_igd:
				min_igd = igd
		sum_igd += min_igd

	return sum_igd