import numpy as np
from genericlayer import GenericLayer, WithNet
import utils


############################### Layer for Sequential ###############################


class LinearLayer(GenericLayer):
    def __init__(self, input_size, output_size, weights ='gaussian', L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        self.input_size = input_size
        self.output_size = output_size
        self.W = utils.SharedWeights.get_or_create(weights, input_size + 1, output_size)

    def forward(self, x, update = False):
        self.x = np.hstack([x, 1])
        return self.W.get().dot(self.x)

    def backward(self, dJdy, optimizer = None):
        dJdx = self.W.get()[:, 0:self.input_size].T.dot(dJdy)
        if optimizer:
            optimizer.update_dW(self.W, self.dJdW_gradient(dJdy))
        return dJdx

    def dJdW_gradient(self, dJdy):
        dJdW = np.multiply(np.matrix(self.x).T, dJdy).T + self.L1 * np.sign(self.W.get()) + self.L2 * self.W.get()
        return dJdW

class MWeightLayer(GenericLayer):
    def __init__(self, input_size, output_size, weights ='gaussian', L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        self.input_size = input_size
        self.output_size = output_size
        self.W = utils.SharedWeights.get_or_create(weights, input_size, output_size)

    def forward(self, x, update = False):
        self.x = x
        return self.W.get().dot(self.x)

    def backward(self, dJdy, optimizer = None):
        dJdx = self.W.get().T.dot(dJdy)
        if optimizer:
            optimizer.update_dW(self.W, self.dJdW_gradient(dJdy))
        return dJdx

    def dJdW_gradient(self, dJdy):
        dJdW = np.multiply(np.matrix(self.x).T, dJdy).T + self.L1 * np.sign(self.W.get()) + self.L2 * self.W.get()
        return dJdW

class VWeightLayer(GenericLayer):
    def __init__(self, size, weights ='gaussian', L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        self.size = size
        self.W = utils.SharedWeights.get_or_create(weights, 1, size)

    def forward(self, x, update = False):
        self.x = x
        return self.W.get()

    def backward(self, dJdy, optimizer = None):
        if optimizer:
            optimizer.update_dW(self.W, self.dJdW_gradient(dJdy))

        if type(self.x) is list:
            return [np.zeros_like(self.x[ind]) for ind in range(len(self.x))]
        else:
            return np.zeros_like(self.x)

    def dJdW_gradient(self, dJdy):
        return dJdy + self.L1 * np.sign(self.W.get()) + self.L2 * self.W.get()

class Lock(GenericLayer):
    def __init__(self, net):
        self.net = net

    def forward(self, x, update = False):
        return self.net.forward(x)

    def backward(self, dJdy, optimizer = None):
        return self.net.backward(dJdy)

class SoftMaxLayer(GenericLayer):
    def forward(self, x, update = False):
        # print 'xS'+str(x)
        exp_x = np.exp(x-np.max(x))
        # print 'exp_x'+str(exp_x)
        self.y = exp_x/np.sum(exp_x)
        return self.y

    def backward(self, dJdy, optimizer = None):
        dJdx = np.zeros(dJdy.size)
        for i in range(self.y.size):
            aux_y = -self.y.copy()
            aux_y[i] = (1-self.y[i])
            dJdx[i] = self.y[i]*aux_y.dot(dJdy)
        return dJdx

class HeavisideLayer(GenericLayer):
    def forward(self, x, update = False):
        self.y = (x >= 0)*1.0
        return self.y

    def backward(self, dJdy, optimizer = None):
        return dJdy

class SignLayer(GenericLayer):
    def forward(self, x, update = False):
        self.y = np.sign(x)
        return self.y

    def backward(self, dJdy, optimizer = None):
        return dJdy

class TanhLayer(GenericLayer):
    def forward(self, x, update = False):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dJdy, optimizer = None):
        # print 'tanh'+str(self.y)
        return (1.-self.y ** 2) * dJdy

class SigmoidLayer(GenericLayer):
    def forward(self, x, update = False):
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, dJdy, optimizer = None):
        return self.y*(1-self.y)*dJdy

class ReluLayer(GenericLayer):
    def forward(self, x, update = False):
        self.x = x
        return np.maximum(0,x)

    def backward(self, dJdy, optimizer = None):
        return np.maximum(0,self.x > 0)*dJdy

class NegativeLayer(GenericLayer):
    def forward(self, x, update = False):
        return -x

    def backward(self, dJdy, optimizer = None):
        return -dJdy

class SumLayer(GenericLayer):
    def forward(self, x, update = False):
        self.x = np.array(x)
        return np.sum(self.x,0)

    def backward(self, dJdy, optimizer = None):
        return np.array([np.ones(element.size)*dJdy for element in self.x])

class MulLayer(GenericLayer):
    def forward(self, x, update = False):
        self.x = np.array(x)
        return np.prod(self.x,0)

    def backward(self, dJdy, optimizer = None):
        dJdx = []
        for i in range(self.x.shape[0]):
            dJdx.append(np.prod(np.delete(self.x,i,0),0))
        return np.array([element*dJdy for element in dJdx])

class NormalizationLayer(GenericLayer):
    def __init__(self, min_in, max_in, min_out = 0, max_out = 1):
        self.min_in = min_in
        self.max_in = max_in
        self.min_out = min_out
        self.max_out = max_out

    def forward(self, x, update = False):
        return (x-self.min_in)/(self.max_in-self.min_in)*(self.max_out-self.min_out)+self.min_out

    def backward(self, dJdy, optimizer = None):
        return dJdy*(self.max_out-self.min_out)/(self.max_in-self.min_in)

class RandomGaussianLayer(GenericLayer):
    def __init__(self, sigma = 1):
        self.sigma = sigma

    def forward(self, x, update = False):
        if update == True:
            self.y = x + np.random.normal(0,self.sigma,size=x.size)
        else:
            self.y = x
        return self.y

    def backward(self, dJdy, optimizer = None):
        return dJdy

#############################################################################################
############################### Layer for Computational Graph ###############################

class ComputationalGraphLayer(WithNet):
    def __init__(self, operation):
        WithNet.__init__(self,operation.get())

class SelectVariableLayer(GenericLayer):
    def __init__(self, variables, variable):
        self.variables = variables
        variables_dict = {}
        for ind, var in enumerate(variables):
            variables_dict[var] = ind

        self.ind = variables_dict[variable]

    def forward(self, x_group, update = False):
        self.x = x_group
        # print 'select var '+str(x_group)
        if type(x_group) is list:
            return x_group[self.ind]
        else:
            return x_group

    def backward(self, dJdy, optimizer = None):
        # print 'SelectVariableLayer'
        # print dJdy
        if len(self.variables) == 1:
            return dJdy
        else:
            # print [dJdy if ind == self.ind else np.zeros(self.x[ind].shape) for ind,var in enumerate(self.variables)]
            # print 'exit'
            return [dJdy if ind == self.ind else np.zeros(self.x[ind].shape) for ind,var in enumerate(self.variables)]

class VariableDictLayer(GenericLayer):
    def __init__(self, variables):
        self.variables = variables

    def forward(self, x_dict, update = False):
        self.x = x_dict
        return [self.x[var] for var in self.variables]

    def backward(self, dJdy, optimizer = None):
        dJdx = {}
        for var,dJdvar in zip(self.variables,dJdy):
            dJdx[var] = dJdvar
        return dJdx

class ConstantLayer(GenericLayer):
    def __init__(self, value):
        self.value = value

    def forward(self, x, update = False):
        self.x = np.array(x)
        return self.value

    def backward(self, dJdy, optimizer = None):
        return np.zeros(self.x.shape)

class ConcatLayer(GenericLayer):
    def forward(self, x, update = False):
        self.x = x
        return np.hstack(x)

    def backward(self, dJdy, optimizer = None):
        inds = np.cumsum(map(len,self.x))
        return np.split(dJdy,inds)[:-1]