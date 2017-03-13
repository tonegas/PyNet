import numpy as np
from itertools import izip

class Trainer():
    def __init__(self, depth = 1):
        self.depth = depth

    def train(self, model, dJdy, optimizer, depth = None):
        if depth == None:
            depth = self.depth

        aux_dJdy = dJdy
        if model.group:
            if hasattr(model,'layers'):
                for layer in reversed(model.layers):
                    aux_dJdx = layer.backward(aux_dJdy)
                    if hasattr(layer,'dJdW_gradient') and hasattr(optimizer, 'update'):
                        optimizer.update(layer, layer.dJdW_gradient(aux_dJdy))
                    elif depth-1 > 0:
                        self.train(layer, aux_dJdy, optimizer, depth-1)
                    aux_dJdy = aux_dJdx
            else:
                aux_dJdx = []
                a = 0
                for ind,element in enumerate(model.elements):
                    b = a+model.vect_size[ind]
                    aux_dJdx.append(element.backward(aux_dJdy[a:b]))
                    if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                        optimizer.update(element, element.dJdW_gradient(aux_dJdy[a:b]))
                    elif depth-1 > 0:
                        self.train(element, aux_dJdy[a:b], optimizer, depth-1)
                    a = b
                aux_dJdy = np.sum(np.array(aux_dJdx),0)
        else:
            aux_dJdx = model.backward(aux_dJdy)
            if hasattr(model,'dJdW_gradient') and hasattr(optimizer, 'update'):
                optimizer.update(model, model.dJdW_gradient(aux_dJdy))
            aux_dJdy = aux_dJdx

        return aux_dJdy

    def learn_one(self, model, input, target, loss, optimizer):
        y = model.forward(input)
        J = loss.loss(y,target)
        dJdy = loss.dJdy_gradient(y,target)
        self.train(model, dJdy, optimizer)
        return J, dJdy

    def learn(self, model, input_data, target_data, loss, optimizer, epochs):
        J_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        num_sample = len(input_data)
        for epoch in range(epochs):
            for x,t in izip(input_data,target_data):
                J, dJdy = self.learn_one(model, x, t, loss, optimizer)
                J_list[epoch] += np.sqrt(np.sum(np.power(J,2)))/num_sample
                dJdy_list[epoch] = np.mean(dJdy)

        return J_list, dJdy_list
#