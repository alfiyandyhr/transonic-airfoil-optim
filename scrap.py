import numpy as np
import pandas as pd

def scrap_initial_training_data():
	"""Scraping the data of interest from CFD simulation results as objective functions"""
	data_interest_obj = [] #for objective functions
	for i in range(50):
		data = pd.read_csv('Meshes/random_design'+str(i)+'/history.csv',
			header=0, usecols=['      "CEff"      '])
		interest = np.array([data.iloc[len(data)-1,0]])
		data_interest_obj = np.append(data_interest_obj, interest)

	"""Scraping the data of interest from random designs as design variables"""
	data_interest_dv = np.zeros((1,28)) #for design variables
	for i in range(50):
		data_upper = np.genfromtxt('Designs/initial_samples/control_points/random_design'
			+ str(i) + '.dat', skip_header=1, skip_footer=17,
			usecols=(1), delimiter=' ')
		data_lower = np.genfromtxt('Designs/initial_samples/control_points/random_design'
			+ str(i) + '.dat', skip_header=17, skip_footer=1,
			usecols=(1), delimiter=' ')
		data_upper = np.array([data_upper])
		data_lower = np.array([data_lower])
		interest = np.append(data_upper, data_lower, axis=1)
		data_interest_dv = np.append(data_interest_dv, interest, axis=0)
	data_interest_dv = np.delete(data_interest_dv, 0, 0)

	"""Saving to dat files"""
	np.savetxt('Data/Training/X.dat',
				data_interest_dv,
	 			delimiter=' ',
	 			header=None,
	 			footer='')
	np.savetxt('Data/Training/OUT.dat',
				data_interest_obj,
	 			delimiter=' ',
	 			header=None,
	 			footer='')