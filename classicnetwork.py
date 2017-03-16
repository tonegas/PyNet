import numpy as np

from genericlayer import GenericLayer
from layers import LinearLayer, SignLayer
from network import Sequential

class Hopfield(GenericLayer):
    def __init__(self, state_size):
        self.state_size = state_size
        self.net = Sequential([
            LinearLayer(state_size, state_size, weights = np.zeros([state_size, state_size + 1])),
            SignLayer()
        ])

    def step(self, x):
        pass

    def forward(self, x):
        y = self.net.forward(x)
        diff = y!=x
        while sum(diff):
            min_energy = -1/2*x.dot(self.net.layers[0].W).dot(np.hstack([x,1]))
            min_energy_ind = None
            for i,changed in enumerate(diff):
                if changed:
                    x_aux = x.copy()
                    x_aux[i] = y[i]
                    min_energy_aux = -1/2*x_aux.dot(self.net.layers[0].W).dot(np.hstack([x_aux,1]))
                    if min_energy_aux < min_energy:
                        min_energy_ind = i

            if min_energy_ind != None:
                x[min_energy_ind] = y[min_energy_ind]
            else:
                x = y

            y = self.net.forward(x)
            diff = y!=x

        return y

    def save_state(self, x):
        self.net.layers[0].W += x.reshape([self.state_size,1]).dot(np.hstack([x,0]).reshape([self.state_size+1,1]).T)-np.eye(self.state_size,self.state_size+1)




