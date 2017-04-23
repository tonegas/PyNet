import numpy as np

import genericlayer
import network
import layers
import utils
import computationalgraph as cp


class AutoEncoder(genericlayer.GenericLayer):
    def __init__(self, input_size, layers_params):
        self.elements = []
        self.layers_params = layers_params
        self.layers_sizes = [input_size]
        map(self.layers_sizes.append,[layer_param['size'] for layer_param in layers_params])

        x = cp.Input(['x'],'x')
        for ind, layer_param in enumerate(layers_params):
            #print (self.layers_sizes[ind],self.layers_sizes[ind+1])
            W = cp.MWeight(
                    self.layers_sizes[ind],
                    self.layers_sizes[ind+1],
                    weights = layer_param.get('weights','gaussian'),
                    L1 = layer_param.get('L1',0.0),
                    L2 = layer_param.get('L2',0.0)
                )
            b = cp.VWeight(
                    self.layers_sizes[ind+1],
                    weights = layer_param.get('biases','gaussian'),
                    L1 = layer_param.get('L1',0.0),
                    L2 = layer_param.get('L2',0.0)
                )
            self.elements.append(
                layers.ComputationalGraphLayer(W*x+b)
            )

    def choose_network(self, ind_used_layers = None, ind_locked_layers=[]):
        self.model = network.Sequential()
        ind_used_layers = range(len(self.layers_params)) if ind_used_layers == None else ind_used_layers
        for ind in ind_used_layers:
            if ind in ind_locked_layers:
                self.model.add(
                    layers.Lock(self.elements[ind]),
                )
            else:
                self.model.add(
                    self.elements[ind]
                )
            self.model.add(self.layers_params[ind]['output_layer'])

    def forward(self, x, update = False):
        return self.model.forward(x)

    def backward(self, dJdy, optimizer = None):
        return self.model.backward(dJdy, optimizer)

ae = AutoEncoder(10,[
    {'size' : 7, 'output_layer' : layers.SigmoidLayer},
    {'size' : 5, 'output_layer' : layers.SigmoidLayer},
    {'size' : 3, 'output_layer' : layers.SigmoidLayer},
    {'size' : 5, 'output_layer' : layers.SigmoidLayer},
    {'size' : 7, 'output_layer' : layers.SigmoidLayer},
    {'size' : 10, 'output_layer' : layers.SigmoidLayer},
])
ae.choose_network()
print ae.forward(np.array([1,2,3,4,5,6,7,8,9,10]))
print ae.backward(np.array([1,2,3,4,5,6,7,8,9,10]))