import numpy as np

from genericlayer import GenericLayer
from layers import LinearLayer, SignLayer, SigmoidLayer, TanhLayer, SumLayer, MulLayer
from network import Sequential, SumGroup, MulGroup, ParallelGroup

class Hopfield(GenericLayer):
    def __init__(self, state_size):
        self.state_size = state_size
        self.net = Sequential([
            LinearLayer(state_size, state_size, weights = 'zeros'),
            SignLayer()
        ])

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

    def store(self, x):
        self.net.layers[0].W += x.reshape([self.state_size,1]).dot(np.hstack([x,0]).reshape([self.state_size+1,1]).T)-np.eye(self.state_size,self.state_size+1)

class Kohonen(GenericLayer):
    def __init__(self, input_size, output_size, topology, weights = 'random', learning_rate = 0.1, radius = 0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.radius = radius
        if type(topology) == tuple:
            if (len(topology) == 2 and type(topology[0]) == int and type(topology[1]) == bool):
                self.positions = [x for x in range(topology[0])]
                self.topology = (topology[0])
                self.topology_close = type(topology[1])
            elif (len(topology) == 3 and
                    (type(topology[0] == int and type(topology[1]) == int and topology[0]*topology[1] == output_size) and
                    type(topology[2]) == bool)):
                self.positions = [(x,y) for x in range(topology[0]) for y in range(topology[1])]
                self.topology = (topology[0],topology[1])
                self.topology_close = type(topology[2])
        else:
            raise Exception('Type not correct!')

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

    def best_matching_unit(self, x):
        return np.argmin(np.sum((np.array(self.W)-x)**2,1))

    def phi(self, d):
        return np.maximum(1-(d**2/self.radius**2),0)

    def distance(self, winner):
        if self.topology_close:
            return np.sqrt(np.sum(np.min([np.abs(np.array(self.positions)-self.positions[winner]),self.topology - np.abs(np.array(self.positions)-self.positions[winner])],0)**2,1))
        else:
            return np.sqrt(np.sum((np.array(self.positions)-self.positions[winner])**2,1))

    def forward(self, x):
        return self.W[self.best_matching_unit(x),:]

    def forward_and_update(self, x):
        best_matching_unit = self.best_matching_unit(x)
        d = self.distance(best_matching_unit)
        self.W = self.learning_rate*np.array([self.phi(d)]).T*(np.array([x]).repeat(self.output_size)-self.W)

class Vanilla(GenericLayer):
    def __init__(self, input_size, output_size,  memory_size):
        self.n1 = SumGroup(LinearLayer(input_size,memory_size),LinearLayer(memory_size,memory_size))
        self.n2 = LinearLayer(memory_size,output_size)
        self.h = np.zeros(memory_size)


    def forward(self, x):
        self.h = self.n1.forward([x,self.h])
        return self.n2.forward(self.h)

    def backward(self, dJdy):
        dJdx = self.n2.backward(dJdy)
        return self.n1.backward(dJdx)

    def backward_and_update(self, dJdy, optimizer, depth):
        dJdx = self.n2.backward_and_update(dJdy, optimizer, depth)
        return self.n1.backward_and_update(dJdx, optimizer, depth)

class LSTM(GenericLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.n1 = SumGroup(
            MulGroup(Sequential(LinearLayer(input_size+output_size,output_size),SigmoidLayer),GenericLayer),
            MulGroup(Sequential(LinearLayer(input_size+output_size,output_size),SigmoidLayer),Sequential(LinearLayer(input_size+output_size,output_size),TanhLayer))
        )
        self.n2 = MulGroup(
            Sequential(GenericLayer, TanhLayer),
            Sequential(LinearLayer(input_size+output_size, output_size),SigmoidLayer)
        )
        self.ct = np.zeros(output_size)
        self.ht = np.zeros(output_size)

    def forward(self, x):
        self.ct = self.n1.forward([[np.append(x,self.ht),self.ct],[np.append(x,self.ht),np.append(x,self.ht)]])
        self.ht = self.n2.forward([self.ct,np.append(x,self.ht)])
        return self.ht

    def backward(self, dJdy):
        dJdx_group = self.n2.backward(dJdy)
        [[dJdx1,dJdx2],[dJdx3,dJdx4]] = self.n1.backward(dJdx_group[0])
        dJdx = dJdx_group[1]+dJdx1+dJdx3+dJdx4
        return dJdx[:self.input_size]

    def backward_and_update(self, dJdy, optimizer, depth):
        dJdx_group = self.n2.backward_and_update(dJdy, optimizer, depth)
        [[dJdx1,dJdx2],[dJdx3,dJdx4]] = self.n1.backward_and_update(dJdx_group[0], optimizer, depth)
        dJdx = dJdx_group[1]+dJdx1+dJdx3+dJdx4
        return dJdx[:self.input_size]