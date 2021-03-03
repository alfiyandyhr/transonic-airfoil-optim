#KMeans clustering algorithm in pytorch
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/17/2021
#####################################################################################################
import torch
import math
import matplotlib.pyplot as plt

class KMeans:
	"""
	Implementation of KMeans clustering in pytorch
	Parameters:
		n_clusters	: number of clusters (int)
		max_iter	: maximum number of iteration (int)
		tol			: tolerance (float)
		mode		: type of distance measure {'euclidean','cosine'}
		minibatch	: batchsize of MinibatchKMeans {None, int}
	Attributes:
		centroids	: cluster centroids (torch.Tensor)
	"""
	def __init__(self,
				 n_clusters,
				 max_iter=100,
				 tol=0.0001,
				 mode='euclidean',
				 minibatch=None,
				 verbose=0):

		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.mode = mode
		self.minibatch = minibatch
		self._loop = False
		self._show = False
		self.centroids = None
		self.verbose = verbose

		try:
			import PYNVML
			self._pynvml_exist = True
		except ModuleNotFoundError:
			self._pynvml_exist = False

	@staticmethod
	def cosine_similarity(a, b):
		"""
		Compute cosine similarity of 2 sets of vectors
		Parameters:
			a: torch.Tensor, shape: [m, n_features]
			b: torch.Tensor, shape: [n, n_features]
		"""

		a_norm = a.norm(dim=-1, keepdim=True)
		b_norm = b.norm(dim=-1, keepdim=True)
		a = a / (a_norm + 1e-8)
		b = b / (b_norm + 1e-8)
		return a @ b.transpose(-2,-1)

	@staticmethod
	def euclidean_similarity(a, b):
		"""
		Compute euclidean similarity of 2 sets of vectors
		Parameters:
			a: torch.Tensor, shape: [m, n_features]
			b: torch.Tensor, shape: [n, n_features]
		"""

		return 2 * a @ b.transpose(-2,-1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[...,None,:]

	def remaining_memory(self):
		"""
		Get the remaining memory in GPU
		"""
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
		if self._pynvml_exist:
			pynvml.nvmlInit()
			gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
			info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
			remaining = info.free
		else:
			remaining = torch.cuda.memory_allocated()
		return remaining

	def maximum_similarity(self, a, b):
		"""
		Compute maximum similarity or minimum distance of each vector
		in 'a' with all vectors in 'b'
		Parameters:
			a: torch.Tensor, shape: [m, n_features]
			b: torch.Tensor, shape: [n, n_features]
		"""

		device = a.device.type
		batchsize = a.shape[0]
		if self.mode == 'cosine':
			similarity_function = self.cosine_similarity
		elif self.mode == 'euclidean':
			similarity_function = self.euclidean_similarity

		if device == 'cpu':
			sim = similarity_function(a, b)
			max_sim_v, max_sim_i = sim.max(dim=-1)
			return max_sim_v, max_sim_i
		else:
			if a.dtype == torch.float:
				expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
			elif a.dtype == torch.half:
				expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
			ratio = math.ceil(expected / self.remaining_memory())
			sub_batchsize = math.ceil(batchsize/ratio)
			msv, msi = [], []
			for i in range(ratio):
				if i*sub_batchsize >= batchsize:
					continue
				sub_x = a[i*sub_batchsize: (i+1)*sub_batchsize]
				sub_sim = similarity_function(sub_x, b)
				sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
				del sub_sim
				msv.append(sub_max_sim_v)
				msi.append(sub_max_sim_i)
			if ratio == 1:
				max_sim_v, max_sim_i = msv[0], msi[0]
			else:
				max_sim_v = torch.cat(msv, dim=0)
				max_sim_i = torch.cat(msi, dim=0)

			return max_sim_v, max_sim_i

	def fit_predict(self, X, centroids=None):
		"""
		Combination of fit() and predict() methods.
		Much faster than calling fit() and predict() separately
		Parameters:
			X: torch.Tensor, shape: [n_samples, n_features]

			centroids: {torch.Tensor, None}, default: None
				if given, centroids will be initilized with given tensor
				if None, centroids will be randomly chosen from X

		Return:
			labels: torch.Tensor, shape: [n_samples]	
		"""

		batchsize, emb_dim = X.shape
		device = X.device.type

		if centroids is None:
			self.centroids = X[torch.randint(low=0,high=99,size=(self.n_clusters,),device=device)]
		
		else:
			self.centroids = centroids

		num_points_in_clusters = torch.ones(self.n_clusters, device=device)

		closest = None

		for i in range(self.max_iter):
			if self.minibatch is not None:
				x = X[torch.randint(low=0,high=batchsize,size=(self.minibatch,),device=device)]
			else:
				x = X

			closest = self.maximum_similarity(a=x, b=self.centroids)[1]
			matched_clusters, counts = closest.unique(return_counts=True)

			c_grad = torch.zeros_like(self.centroids)
			if self._loop:
				for j, count in zip(matched_clusters, counts):
					c_grad[j] = x[closest==j].sum(dim=0) / count
			else:
				if self.minibatch is None:
					expanded_closest = closest[None].expand(self.n_clusters, -1)
					mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[:, None]).float()
					c_grad = mask @ x /mask.sum(-1)[..., :, None]
					c_grad [c_grad!=c_grad] = 0
				else:
					expanded_closest = closest[None].expand(len(matched_clusters), -1)
					mask = (expanded_closest==matched_clusters[:, None]).float()
					c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]

			error = (c_grad - self.centroids).pow(2).sum()
			if self.minibatch is not None:
				lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
				# lr = 1/num_points_in_clusters[:,None] ** 0.1
			else:
				lr = 1

			num_points_in_clusters[matched_clusters] += counts
			self.centroids = self.centroids * (1-lr) + c_grad * lr
			if self.verbose >= 2:
				print('iter:', i, 'error:', error.item())
			if error <= self.tol:
				break

		#Scatter
		if self._show:
			if self.mode == 'cosine':
				sim = self.cosine_similarity(x, self.centroids)
			elif self.mode == 'euclidean':
				sim = self.euclidean_similarity(x, self.centroids)
			closest = sim.argmax(dim=-1)
			plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=closest.cpu(), marker='.', cmap='hsv')
			plt.show()

		#End scatter
		if self.verbose >=1:
			print(f'used {i+1} iterations to cluster {batchsize} items into {self.n_clusters} clusters\n')
			print('------------------------------------------------------------------')
		return closest

	def predict(self, X):
		"""
		Predict the closest cluster each sample in X belongs to
		Parameters:
			X: torch.Tensor, shape: [n_samples, n_features]

		Return:
			labels: torch.Tensor, shape: [n_samples]
		"""
		return self.maximum_similarity(a=X, b=self.centroids)[1]

	def fit(self, X):
		"""
		Perform KMeans clustering
		Parameters:
			X: torch.Tensor, shape: [n_samples, n_features]
		"""
		self.fit_predict(X, centroids)

	def inertia_(self, X, labels):
		"""
		Calculating the mean squared distance of X from the centroids
		Parameters:
			X: torch.Tensor, shape: [n_samples, n_features]
			labels: torch.Tensor, shape: [n_samples]
		Return:
			inertia: the mean squared distance
		"""
		device = X.device.type
		inertia = torch.tensor(0.0).to(device)
		for i in range(len(X)):
			inertia += torch.sqrt(torch.sum(torch.pow((X[i] - self.centroids[labels[i]]),2)))

		return inertia/len(X)