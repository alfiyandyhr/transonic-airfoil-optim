import numpy as np
from AirfoilDesign import *
import matplotlib.pyplot as plt

def calc_area(array):
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
	for i in range(int(len(array)/2)-1):
		h1 = array[i,1] - array[len(array)-1-(i+1),1]
		h2 = array[i+1,1] - array[len(array)-1-(i+1),1]
		area += 0.5 * (array[i+1,0]-array[i,0]) * max(h1,h2)

	#Calculate the lower triangles
	for i in range(int(len(array)/2),len(array)-1):
		h1 = array[i,1] - array[len(array)-1-(i+1),1]
		h2 = array[i+1,1] - array[len(array)-1-(i+1),1]
		area += 0.5 * (array[i+1,0]-array[i,0]) * min(h1,h2)

	return area