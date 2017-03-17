import numpy as np
import matplotlib.pyplot as plt

from classicnetwork import Hopfield, FullyConnected
from layers import  LinearLayer, SigmoidLayer, SignLayer,TanhLayer
from network import Sequential

# h = Hopfield(100)
# zz = np.ones(100)
# zz[50] = -1
# h.save_state(zz)
# y = h.forward(np.sign(np.random.rand(100)-0.5))
# print y

n = FullyConnected(Sequential([
    LinearLayer(20,20,weights='zeros'),
    SignLayer()
]),20)
# y=n.forward(np.array([-1,-2,1,4,-5]))
# print y
# y=n.forward(np.array([1,2,11,4,-5]))
# print y

n.save_state(np.array([-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1]))
# n.save_state(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))
# n.save_state(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))
#
n.save_state(np.array([1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1]))
# n.save_state(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))
# n.save_state(np.array([0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]))

n.save_state(np.array([1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1]))

y=np.sign(n.forward(np.array([1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1])))
print y