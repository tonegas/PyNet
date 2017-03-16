import numpy as np
import matplotlib.pyplot as plt

from classicnetwork import Hopfield

h = Hopfield(100)
zz = np.ones(100)
zz[50] = -1
h.save_state(zz)
y = h.forward(np.sign(np.random.rand(100)-0.5))
print y