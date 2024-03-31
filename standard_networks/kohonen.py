import numpy as np
from genericlayer import GenericLayer
import utils


class Kohonen(GenericLayer):
    def __init__(self, input_size, output_size, topology, output_type = 'btu', weights = 'random', learning_rate = 0.1, radius = 0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.radius = radius
        self.output_types = {
            'btu': lambda: self.btu,
            'distance': lambda: self.weight_distance,
            'weight': lambda: self.selected_weight
        }
        self.output_type = self.output_types.get(output_type)
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
        self.W = utils.define_weights(weights, input_size, output_size)
        self.btu = None
        self.weight_distance = None
        self.selected_weight = None

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
        self.btu = self.best_matching_unit(x)
        self.weight_distance = self.distance(self.btu)
        if update:
            self.W += self.learning_rate*np.array([self.phi(self.weight_distance)]).T*(np.array([x]).repeat(self.output_size,0)-self.W)
        self.selected_weight = self.W[self.btu,:]
        return self.output_type()