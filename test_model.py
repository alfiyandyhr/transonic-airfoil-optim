import torch
import numpy as np
from eval import evaluate
from pymoo.model.population import pop_from_array_or_individual
import matplotlib.pyplot as plt
from performance import calc_hv

Model = torch.load('Data/Prediction/trained_model.pth')

pop_1 = np.genfromtxt('Initial_Sampling/dv_lhs.dat')
pop_2 = np.genfromtxt('Designs/gen_2/dv_2.dat')

pop_1_t = torch.from_numpy(pop_1)
pop_2_t = torch.from_numpy(pop_2)

pop_1 = pop_from_array_or_individual(pop_1)

pop_eval_1 = evaluate(gen=1, pop=pop_1, mesh=False, CFD=False,
					  done = [-1], diverge=[-1])

pop_eval_1_pred = Model(pop_1_t.float()).detach().numpy()
pop_eval_2_pred = Model(pop_2_t.float()).detach().numpy()

plt.plot(pop_eval_1[:,0],pop_eval_1[:,1],'o')
plt.plot(pop_eval_1_pred[:,0],pop_eval_1_pred[:,1],'ro')
plt.show()