import numpy as np

from genericlayer import GenericLayer
from layers import define_weights, LinearLayer, SignLayer, SigmoidLayer, TanhLayer, SumLayer, MulLayer, ComputationalGraphLayer
from network import Sequential, SumGroup, MulGroup, ParallelGroup


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

class Kohonen(GenericLayer):
    def __init__(self, input_size, output_size, topology, weights = 'random', learning_rate = 0.1, radius = 0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.radius = radius
        if type(topology) == tuple:
            if (len(topology) == 2 and type(topology[0]) == int and type(topology[1]) == bool):
                self.positions = [x for x in range(topology[0])]
                self.topology = (topology[0])
                self.topology_close = topology[1]
            elif (len(topology) == 3 and
                    (type(topology[0] == int and type(topology[1]) == int and topology[0]*topology[1] == output_size) and
                    type(topology[2]) == bool)):
                self.positions = [(x,y) for x in range(topology[0]) for y in range(topology[1])]
                self.topology = (topology[0],topology[1])
                self.topology_close = topology[2]
        else:
            raise Exception('Type not correct!')

        self.learning_rate = learning_rate
        self.W = define_weights(weights, input_size, output_size)

    def best_matching_unit(self, x):
        return np.argmin(np.sum((np.array(self.W)-x)**2,1))

    def phi(self, d):
        return np.maximum(1.0-((d**2.0)/(self.radius**2.0)),0.0)

    def distance(self, best_matching_unit):
        if self.topology_close:
            return np.sqrt(np.sum(np.min([np.abs(np.array(self.positions)-self.positions[best_matching_unit]),self.topology - np.abs(np.array(self.positions)-self.positions[best_matching_unit])],0)**2,1))
        else:
            return np.sqrt(np.sum((np.array(self.positions)-self.positions[best_matching_unit])**2,1))

    def forward(self, x, update = False):
        if update:
            best_matching_unit = self.best_matching_unit(x)
            d = self.distance(best_matching_unit)
            self.W += self.learning_rate*np.array([self.phi(d)]).T*(np.array([x]).repeat(self.output_size,0)-self.W)
            return self.W[best_matching_unit,:]
        else:
            best_matching_unit = self.best_matching_unit(x)
            d = self.distance(best_matching_unit)
            return self.W[self.best_matching_unit(x),:]

class Vanilla(GenericLayer):
    def __init__(self, input_size, output_size,  memory_size):
        self.n1 = SumGroup(LinearLayer(input_size,memory_size),LinearLayer(memory_size,memory_size))
        self.n2 = LinearLayer(memory_size,output_size)
        self.h = np.zeros(memory_size)

    def forward(self, x, update = False):
        self.h = self.n1.forward([x,self.h])
        return self.n2.forward(self.h)

    def backward(self, dJdy, optimizer = None):
        dJdx = self.n2.backward(dJdy, optimizer)
        return self.n1.backward(dJdx, optimizer)

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
        # Ct = ComputationalGraphLayer(
        #     Sigmoid(Wf*xh+bf)*Ct1+
        #     Sigmoid(Wi*xh+bi)*Tanh(Wc*xh+bc)
        # )

        self.ct = np.zeros(output_size)
        self.ht = np.zeros(output_size)

    def forward(self, x, update = False):
        self.ct = self.n1.forward([[np.append(x,self.ht),self.ct],[np.append(x,self.ht),np.append(x,self.ht)]])
        self.ht = self.n2.forward([self.ct,np.append(x,self.ht)])
        return self.ht

    def backward(self, dJdy,  optimizer = None):
        dJdx_group = self.n2.backward(dJdy, optimizer)
        [[dJdx1,dJdx2],[dJdx3,dJdx4]] = self.n1.backward(dJdx_group[0], optimizer)
        dJdx = dJdx_group[1]+dJdx1+dJdx3+dJdx4
        return dJdx[:self.input_size]