import numpy as np
from genericlayer import GenericLayer
from utils import define_weights

class Hopfield(GenericLayer):
    def __init__(self, state_size):
        self.state_size = state_size
        self.W = define_weights('zeros', state_size, state_size)

    def forward(self, x, update = False):
        y = np.sign(self.W.dot(x))
        diff = y!=x
        while sum(diff):
            min_energy = -1/2*x.dot(self.W).dot(x)
            min_energy_ind = None
            for i,changed in enumerate(diff):
                if changed:
                    x_aux = x.copy()
                    x_aux[i] = y[i]
                    min_energy_aux = -1/2*x_aux.dot(self.W).dot(x_aux)
                    if min_energy_aux < min_energy:
                        min_energy_ind = i

            if min_energy_ind != None:
                x[min_energy_ind] = y[min_energy_ind]
            else:
                x = y

            y = np.sign(self.W.dot(x))
            diff = y!=x

        return y

    def store(self, x):
        self.W += x.reshape([self.state_size,1]).dot(x.reshape([self.state_size,1]).T)-np.eye(self.state_size,self.state_size)
