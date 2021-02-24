#Generating BSpline using geomdl
#Coded by Alfiyandy Hariansyah
#Tohoku University
#1/25/2021
#####################################################################################################
import numpy as np
import pandas as pd
from geomdl import BSpline
from geomdl.visualization import VisMPL
from geomdl import exchange, knotvector
#####################################################################################################
class BSplineFromControlPoints():
	"""Model a bspline from random control points"""
	def __init__(self, degree=3, design_number=50):
		"""Model a bspline with degree 3 by default"""
		self.degree = degree
		self.design_number = design_number

		self.bspline_points = []

	def create(self):
		#Create the curve instance
		crv = BSpline.Curve()
		
		#Set the degree
		crv.degree = self.degree

		for i in range(self.design_number):
			#Set control points
			crv.ctrlpts = exchange.import_txt("Designs/initial_samples/control_points/random_design" + str(i) + ".dat",
			separator = " ")

			#Generate a uniform knotvector
			crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)

			#Get curve points and saving into a file
			self.bspline_points = np.array(crv.evalpts)
			bspline_points_pd = pd.DataFrame(self.bspline_points)

			bspline_points_pd.to_csv('Designs/initial_samples/bspline_points/bspline_points_random' + str(i) + '.dat',
				sep=' ',header=False,index=False)

	def to_pointwise_format(self):
		"""Saving files to pointwise format"""
		for i in range(self.design_number):
			with open('Designs/initial_samples/bspline_points/bspline_points_random' + str(i) + '.dat', 'r') as file_in:
				original_content_upper = file_in.readlines()[0:50]
				file_in.seek(0)	#resetting the pointer into 1st line
				original_content_lower = file_in.readlines()[50:100]
			with open('Designs/initial_samples/bspline_points/bspline_points_random' + str(i) + '.dat', 'w') as file_out:
				file_out.seek(0)
				file_out.write(str(51) + '\n')
				for line1 in original_content_upper:
					file_out.writelines(line1)
				file_out.write(str(1.0) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
				file_out.write(str(51) + '\n')
				file_out.write(str(1.0) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
				for line2 in original_content_lower:
					file_out.writelines(line2)


