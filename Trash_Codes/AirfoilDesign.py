#AirfoilDesign class: import_baseline, create_random, save_random
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/28/2021
#####################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geomdl import BSpline
from geomdl.visualization import VisMPL
from geomdl import exchange, knotvector
#####################################################################################################

class AirfoilDesign:
	"""Model an airfoil design"""
	def __init__(self, design_number, control_points):
		"""Airfoil criteria"""
		self.design_number = design_number
		self.control_points = control_points

		self.uiuc_data_upper = []
		self.uiuc_data_lower = []
		self.control_upper = []
		self.control_lower = []

		self.x_upper = []
		self.y_upper = np.zeros((design_number,int((control_points/2))+2))
		
		self.x_lower = []
		self.y_lower = np.zeros((design_number,int((control_points/2))+1))

		self.y_upper_max = []
		self.y_lower_max = []

		self.y_upper_min = []
		self.y_lower_min = []

		#The max in min in design space
		self.control_max = []
		self.control_min = []

	def import_baseline(self,
		baseline_path,
		baseline_control_path):
		"""Importing the baseline"""
		self.uiuc_data_upper = np.genfromtxt(
					baseline_path,
					usecols = (0,1),
					skip_header = 1,
					skip_footer = 66,
					delimiter = " ")

		self.uiuc_data_lower = np.genfromtxt(
					baseline_path,
					usecols = (0,1),
					skip_header = 67,
					skip_footer = 0,
					delimiter = " ")


		self.control_upper = np.genfromtxt(
					baseline_control_path,
					usecols = (0,1),
					skip_header = int(self.control_points/2)+1,
					skip_footer = 0,
					delimiter = " ")

		self.control_lower = np.genfromtxt(
					baseline_control_path,
					usecols = (0,1),
					skip_header = 0,
					skip_footer = int(self.control_points/2)+2,
					delimiter = " ")

	def create_random(self, sampling_matrix, perturbation=0.02):
		"""Create random coordinates for random samples"""

		self.x_upper = self.control_upper[:,0]
		self.x_lower = self.control_lower[:,0]

		delta_y = perturbation
		
		sm = sampling_matrix

		#The upper part of the airfoil
		for i in range(self.design_number):
			for j in range(int((self.control_points/2))+2):
				if j != 0 and j != int((self.control_points/2))+1:
					#The leading edge
					if j == 1:
						self.y_upper[i,j] = round((self.control_upper[j,1]-delta_y/2)+sm[i,j-1]*2*(delta_y/2),6)
					#The middle section
					elif 1 < j < 12:
						self.y_upper[i,j] = round((self.control_upper[j,1]-delta_y)+sm[i,j-1]*2*(delta_y),6)
					#The trailing edge
					elif j == 12:
						self.y_upper[i,j] = round((self.control_upper[j,1]-delta_y/2)+sm[i,j-1]*2*(delta_y/2),6)
					elif j == 13:
						self.y_upper[i,j] = round((self.control_upper[j,1]-delta_y/10)+sm[i,j-1]*2*(delta_y/10),6)
		
		#The lower part of the airfoil
		for i in range(self.design_number):
			for j in range(int((self.control_points)/2)+1):
				if j != 0:
					#The trailing edge
					if j == 1:
						self.y_lower[i,j] = round((self.control_lower[j,1]-delta_y/10)+sm[i,j+12]*2*(delta_y/10),6)
					elif j == 2:
						self.y_lower[i,j] = round((self.control_lower[j,1]-delta_y/2)+sm[i,j+12]*2*(delta_y/2),6)
					#The middle section
					elif 2 < j < 13:
						self.y_lower[i,j] = round((self.control_lower[j,1]-delta_y)+sm[i,j+12]*2*(delta_y),6)
					#The leading edge
					elif j == 13:
						self.y_lower[i,j] = round((self.control_lower[j,1]-delta_y/2)+sm[i,j+12]*2*(delta_y/2),6)

	def save_random(self):
		"""Saving random coordinates into .dat files"""
		x_upper_pd = pd.DataFrame(self.x_upper)
		x_lower_pd = pd.DataFrame(self.x_lower)
		y_upper_pd = pd.DataFrame(self.y_upper)
		y_lower_pd = pd.DataFrame(self.y_lower)
		z_upper_pd = pd.DataFrame(np.zeros((len(self.x_upper),1)))
		z_lower_pd = pd.DataFrame(np.zeros((len(self.x_lower),1)))


		for i in range(self.design_number):
			random_design_upper = pd.concat([x_upper_pd, y_upper_pd.iloc[i], z_upper_pd], axis=1)
			random_design_lower = pd.concat([x_lower_pd, y_lower_pd.iloc[i], z_lower_pd], axis=1)

			random_design = pd.concat([random_design_lower, random_design_upper], axis=0)

			random_design.to_csv('Designs/initial_samples/control_points/random_design' + str(i) + '.dat',
						sep=' ',header=False,index=False)