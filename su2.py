#CFD using SU2 Solver
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/3/2021
#####################################################################################################
import os

#####################################################################################################

def su2_cfd(dir, paralel_comp, core):
	"""Conducting CFD simulation using SU2 solver"""

	#Change directory
	os.chdir(dir)

	#SU2 simulation with 'core' number of core(s)
	if paralel_comp==True:
		os.system('mpirun -n ' + str(core) + ' SU2_CFD inv_transonic_airfoil.cfg')
	else:
		os.system('SU2_CFD inv_transonic_airfoil.cfg')

	#Return to the parent's directory
	os.chdir('../../')