from pyDOE import *
import matplotlib.pyplot as plt

#criterion arguments: center 'c', maximin 'm', centermaximin 'cm', correlation 'corr'


design = lhs(2,samples=100,criterion=None,iterations=None)

x = design[:,0]
y = design[:,1]

plt.plot(x,y,'o')
plt.show()