import numpy as np

from standart_network.hopfield import Hopfield

n = Hopfield(20)

n.store(np.array([-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1]))
# n.store(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))
# n.store(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))
#
n.store(np.array([1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1]))
# n.store(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))
# n.store(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))

n.store(np.array([1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1]))

y=n.forward(np.array([-1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,-1]))
print y