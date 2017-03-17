import numpy as np

from genericlayer import GenericLayer
from layers import LinearLayer, SignLayer
from network import Sequential

class Hopfield(GenericLayer):
    def __init__(self, state_size):
        self.state_size = state_size
        self.net = Sequential([
            LinearLayer(state_size, state_size, weights = 'zeros'),
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

    def learn(self, x):
        self.net.layers[0].W += x.reshape([self.state_size,1]).dot(np.hstack([x,0]).reshape([self.state_size+1,1]).T)-np.eye(self.state_size,self.state_size+1)

class Kohonen(GenericLayer):
    def __init__(self, input_size, output_size, topology, weights = 'random', learning_rate = 0.1, radius = 0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.radius = radius
        self.topology = topology
        if type(topology) == str:
            if topology == 'line' or topology == 'ring':
                self.position = range(output_size)
            elif topology == 'gird' or topology == 'mesh':
                self.position = [(x,y) for x in range(output_size) for y in range(output_size)]

        self.learning_rate = learning_rate
        if type(weights) == str:
            if weights == 'random':
                self.W = np.random.rand(output_size, input_size)
            elif weights == 'ones':
                self.W = np.ones([output_size, input_size])
            elif weights == 'zeros':
                self.W = np.zeros([output_size, input_size])
        elif type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
            self.W = weights
        else:
            raise Exception('Type not correct!')

    def phi(self, d):
        return np.maximum(1-(d**2/self.radius**2),0)

    def distance(self, winner):
        return np.sqrt(np.sum((np.array(self.position)-self.position[winner])**2,1))

    def forward(self, x):
        y = self.W.dot(self.x)

    def learn(self, x):
        winner = self.winner(x)
        d = self.distance(winner)
        self.W = self.learning_rate*np.array([self.phi(d)]).T*(np.array([x]).repeat(self.output_size)-self.W)


