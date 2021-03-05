import numpy as np
from AirfoilDesign import *
import matplotlib.pyplot as plt

def evaluate_area(array):
	"""
	This function calculates the area of airfoil,
	by summing all the discrete triangles
	Input:
		array of coordinates (x and y)
	Output:
		the area (double)
	"""
	area = 0.0
	#Calculate the upper triangles
	for i in range(int((len(array)+1)/2)-1):
		h1 = array[i,1] - array[len(array)-(i+1),1]
		h2 = array[i+1,1] - array[len(array)-(i+1),1]
		area += 0.5 * (array[i+1,0]-array[i,0]) * max(h1,h2)

	#Calculate the lower triangles
	for i in range(int((len(array)+1)/2),len(array)):
		if i == len(array)-1:
			h1 = array[i,1] - array[0,1]
			h2 = array[0,1] - array[0,1]
			area += 0.5 * (array[0,0]-array[i,0]) * min(h1,h2)
		else:
			h1 = array[i,1] - array[len(array)-(i+1),1]
			h2 = array[i+1,1] - array[len(array)-(i+1),1]
			area += 0.5 * (array[i+1,0]-array[i,0]) * min(h1,h2)

	return area

def evaluate_y_diff(array):
	"""
	This function calculates the difference of y coordinates,
	between each point in the upper and lower coordinates
	Input:
		array of y coordinates (flattened)
	Output:
		the array of y_diffs
	"""
	y_diffs = []
	for i in range(int((len(array)+1)/2)):
		if i==0:
			y_diffs.append(0.0)
		else:
			y_diffs.append(array[i]-array[len(array)-i])

	return(np.array(y_diffs).reshape(int((len(array)+1)/2),1))
