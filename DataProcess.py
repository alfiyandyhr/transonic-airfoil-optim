#Data process includes normalization, clustering, cross validation etc
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/17/2021
#####################################################################################################
import torch
from kmeans import KMeans

def normalize(array, v_max, v_min, axis):
	"""
	This function will output a normalized array
	Input:
		a numpy array or tensor (n x m)
			n = batchsize
			m = parameters for every instance
		v_max = maximum value of each parameter/column
		V_min = minimum value of each parameter/column
	Output:
		a normalized numpy array like input
	"""
	n_rows, n_cols = array.shape

	if axis == 0:
		for row in range(n_rows):
			array[row] = (array[row]-v_min)/(v_max-v_min)
		# array = (array-v_min)/(v_max-v_min)

	if axis == 1:
		for col in range(n_cols):
			array[:,col] = (array[:,col]-v_min[col])/(v_max[col]-v_min[col])

	return array

def denormalize(array, v_max, v_min, axis):
	"""
	This function will output a denormalized array
	Input:
		a numpy array or tensor (n x m)
			n = batchsize
			m = parameters for every instance
		v_max = maximum value of each parameter/column
		V_min = minimum value of each parameter/column
	Output:
		a normalized numpy array like input
	"""
	n_rows, n_cols = array.shape

	if axis == 0:
		for row in range(n_rows):
			array[row] = (v_max-v_min)*array[row] + v_min

	if axis == 1:
		for col in range(n_cols):
			array[:,col] = (v_max[col]-v_min[col])*array[:,col] + v_min[col]

	return array

def remove_duplicates(X, OUT, n_var):
	"""
	This function will remove any individuals with a distance
	in design space shorter than a specified value
	"""
	i = 0
	while 1:
		dist = torch.sqrt((torch.sum(torch.pow((X-X[i,:]), 2.0), axis=1))/n_var)
		dist[i] = 1.0 #make sure the current individual doesnt get deleted
		
		#Location in which the dist is more than 0.001
		idx = torch.where(dist>0.001)[0]

		#Picking the individual at location found above
		X   = torch.index_select(X, 0, idx)
		OUT = torch.index_select(OUT, 0, idx)

		
		if len(X) <= i+1:
			break
		else:
			i = i+1

	return X, OUT

def do_gap_statistics(X, n_var, device):
	"""
	This function uses gap statistics method to calculate the number of clusters
	Input:
		X: Training data for the input layer
		n_var: The number of design variables of the problem
	Output:
		N_cluster: the best number of cluster that maximizes gap
	"""
	
	max_cluster = 30
	trials = 10
	X = X.to(device)
	count = torch.zeros(max_cluster, dtype=torch.int32).to(device)
	X_rnd = torch.randn(len(X), n_var).to(device)

	for trial in range(trials):
		gap 	 = torch.zeros(max_cluster, dtype=torch.float32).to(device)
		gap_diff = torch.zeros(max_cluster, dtype=torch.float32).to(device)
		for cluster in range(max_cluster):
			kmeans       = KMeans(n_clusters=cluster+1, mode='euclidean')
			labels 		 = kmeans.fit_predict(X)
			kmeans_rnd	 = KMeans(n_clusters=cluster+1, mode='euclidean')
			labels_rnd	 = kmeans_rnd.fit_predict(X_rnd)
			gap[cluster] = torch.log(kmeans_rnd.inertia_(X_rnd,labels_rnd)/kmeans.inertia_(X,labels))

			if cluster==0:
				gap_diff[0] = 0.0
			else:
				gap_diff[cluster] = gap[cluster] - gap[cluster-1]
				if gap_diff[cluster] < 0.0:
					break

		count[torch.argmax(gap)] = count[torch.argmax(gap)] + 1

	N_cluster = torch.argmax(count)+1
	#+1 because cluster in the range(max_cluster) starts from zero
	return N_cluster

def do_KMeans_clustering(N_cluster, X, device):
	"""
	This function will use KMeans Clustering method to label training data
	according to its proximity with a cluster
	Input:
		N_cluster: number of cluster estimated by Gap Statistics
		X: Training data for the input layer
	Output:
		cluster_label: label assigned to every point
		over_coef: this will be used in the oversampling method to increase
				   number of points in the less densed cluster region
	"""

	X = X.to(device)

	#Instantiating kmeans object
	kmeans = KMeans(n_clusters=N_cluster, mode='euclidean', verbose=1)
	cluster_label = kmeans.fit_predict(X)

	#Calculating the size of cluster (number of data near the cluster centroid)
	cluster_size = torch.zeros(N_cluster, dtype=torch.int32).to(device)
	for cluster in range(N_cluster):
		cluster_size[cluster] = len(torch.where(cluster_label==cluster)[0])

	over_coef = torch.zeros(N_cluster, dtype=torch.int32).to(device)
	for cluster in range(N_cluster):
		over_coef[cluster] = torch.clone((max(cluster_size))/cluster_size[cluster]).to(device)
		if over_coef[cluster] > 10:
			over_coef[cluster] = 10

	return cluster_label.cpu(), over_coef.cpu()

def do_oversampling(N_cluster,
					cluster_label,
					X, OUT,
					over_coef):
	"""
	This function will use oversampling to prevent from overfitting
	Overfitting can happen when training data got stacked in a very densed region
	Oversampling will then be done in the region where cluster size is small
	Input:
		N_cluster: number of cluster estimated by Gap Statistics
		cluster_label: label assigned to every point
		X: Training data for the input layer
		OUT: Training data for the output layer
		over_coef: this will be used in the oversampling method to increase
				   number of points in the less densed cluster region
	Output:
		X_over: Training data for the input layer that has been oversampled
		OUT_over: Training data for the output layer that has been oversampled
	"""
	X_over	 = torch.clone(X)
	OUT_over = torch.clone(OUT)

	for cluster in range(N_cluster):
		idx = torch.where(cluster_label==cluster)[0]
		X_cluster 	= torch.index_select(X, 0, idx)
		OUT_cluster = torch.index_select(OUT, 0, idx)

		for counter in range(over_coef[cluster]-1):
			X_over	 = torch.vstack((X_over, X_cluster))
			OUT_over = torch.vstack((OUT_over, OUT_cluster))

	return X_over, OUT_over

def calc_batchsize(batchrate,
				   train_ratio,
				   X_over):
	"""
	This function calculates the batchsize which is the size of data
	to be processed at once in the training process
	Input:
		train_ratio: the ratio of the data to be used as training set
		batchrate: the percentage from training data to be processed at once
		X_over: Training data for the input layer that has been oversampled
	Output:
		batchsize: The size of data to be processed at once in the training
		N_all: the size of all data after being oversampled
		N_train: the size of training set
		N_valid: the size of validation set
	"""
	N_all = len(X_over)
	N_train = int(N_all*train_ratio)
	N_valid = N_all - N_train

	batchsize = int(batchrate*N_train/100.0)
	if batchsize < 10:
		batchsize = 10
	elif batchsize > 100:
		batchsize = 100

	return batchsize, N_all, N_train, N_valid


def do_cross_validation(N_all, N_train, X_over, OUT_over):
	"""
	This function will separate the data into training and validation set
	This is done to prevent from overfitting
	Validation set is used to guide the learning process of the training set
	Input:
		N_all: the size of all data after being oversampled
		N_train: the size of training set
		train_ratio: the ratio of the data to be used as training set
		X_over: Training data for the input layer that has been oversampled
		OUT_over: Training data for the output layer that has been oversampled
	Output:
		X_train: Training set for the input layer
		OUT_train: Training set for the output layer
		X_valid: Validation set for the input layer
		OUT_valid: Validation set for the output layer
	"""
	#Initializing random permutation
	rand = torch.randperm(N_all)

	#Separating training set and validation set
	X_train   = X_over[rand[0:N_train]]
	X_valid	  = X_over[rand[N_train:N_all]]
	OUT_train = OUT_over[rand[0:N_train]]
	OUT_valid = OUT_over[rand[N_train:N_all]]

	return X_train, X_valid, OUT_train, OUT_valid