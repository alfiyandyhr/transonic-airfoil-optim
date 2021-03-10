#Generating BSpline using geomdl
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/5/2021
#####################################################################################################
import numpy as np
import pandas as pd
from geomdl import BSpline
from geomdl.visualization import VisMPL
from geomdl import exchange, knotvector
#####################################################################################################
class BSplineFromControlPoints():
	"""Model a bspline from random control points"""
	def __init__(self, degree):
		"""Model a bspline with degree 3 by default"""
		self.degree = degree
		self.bspline_points = []

	def create(self, ctrlpts_path):
		#Create the curve instance
		crv = BSpline.Curve()
		
		#Set the degree
		crv.degree = self.degree

		crv.delta = 0.005

		#Set control points
		crv.ctrlpts = exchange.import_txt(ctrlpts_path, separator = " ")

		#Generate a uniform knotvector
		crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)

		#Get curve points and saving into a file
		self.bspline_points = np.array(crv.evalpts)

	def rotate_and_dilate(self, output_path):

		L = 1.0 - min(self.bspline_points[:,0])

		y_leading_l = self.bspline_points[int(len(self.bspline_points)/2)-1,1]
		y_leading_u = self.bspline_points[int(len(self.bspline_points)/2),1]
		y_leading = (y_leading_u + y_leading_l)/2

		tetha = np.arctan(y_leading/L)

		x = np.copy(self.bspline_points[:,0])
		y = np.copy(self.bspline_points[:,1])

		#Rotation
		x_r = 1.0 + (x - 1.0) * np.cos(-tetha) - y * np.sin(-tetha)
		y_r = (x - 1.0) * np.sin(-tetha) + y * np.cos(-tetha)

		#Dilation
		self.bspline_points[:,0] = (x_r - min(x_r))/L

		bspline_points_pd = pd.DataFrame(self.bspline_points)

		bspline_points_pd.to_csv(output_path,sep=' ',header=False,index=False)

	def to_pointwise_format(self, file_path):
		"""Saving files to pointwise format"""
		data = np.genfromtxt(file_path)

		if data[99,0] > data[100,0]:
			origin = data[100]
			idx_origin = 100
		else:
			origin = data[99]
			idx_origin = 99

		data = np.concatenate((data[0:idx_origin],origin.reshape(1,3),data[idx_origin:199]),axis=0)

		#Trailing edge refinement
		data[0,0] = 1.0
		data[0,1] = 0.0
		data[-1,0] = 1.0
		data[-1,1] = 0.0

		np.savetxt(file_path,data)

		with open(file_path, 'r') as file_in:
			original_content_upper = file_in.readlines()[0:idx_origin+1]
			file_in.seek(0)	#resetting the pointer into 1st line
			original_content_lower = file_in.readlines()[idx_origin+1:200]
		with open(file_path, 'w') as file_out:
			file_out.seek(0)
			if idx_origin == 100:
				file_out.write(str(101) + '\n')
			elif idx_origin == 99:
				file_out.write(str(100) + '\n')
			for line1 in original_content_upper:
				file_out.writelines(line1)
			if idx_origin == 100:
				file_out.write(str(100) + '\n')
			elif idx_origin == 99:
				file_out.write(str(101) + '\n')
			for line2 in original_content_lower:
				file_out.writelines(line2)
		

