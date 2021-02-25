import numpy as np

def save(f, array,
		 delimiter=' ',
		 newline='\n',
		 header='',
		 footer='',
		 comments='#'):
	"""
	This function is responsible for any task which includes saving into files
	for example:
		>saving training data
		>saving all_pop_X.dat
		>saving best_pop_X.dat
		etc 
	"""

	np.savetxt(f, array,
			   delimiter=delimiter,
			   newline=newline,
			   header=header,
			   footer=footer,
			   comments=comments)