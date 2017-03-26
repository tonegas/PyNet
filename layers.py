import numpy as np
from genericlayer import GenericLayer

def define_weights(weights, input_size, output_size):
    if type(weights) == str:
        if weights == 'random':
            return np.random.rand(output_size, input_size)
        elif weights == 'norm_random':
            return (np.random.rand(output_size, input_size)-0.5)/input_size
        elif weights == 'ones':
            return np.ones([output_size, input_size])
        elif weights == 'zeros':
            return np.zeros([output_size, input_size])
        else:
            raise Exception('Type not correct!')
    elif type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
        return weights
    else:
        raise Exception('Type not correct!')

class LinearLayer(GenericLayer):
    def __init__(self, input_size, output_size, weights ='random', L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        self.input_size = input_size
        self.output_size = output_size
        self.dW = 0
        self.W = define_weights(weights, input_size + 1, output_size)

    def forward(self, x, update = False):
        self.x = np.hstack([x, 1])
        return self.W.dot(self.x)

    def backward(self, dJdy, optimizer = None):
        dJdx = self.W[:, 0:self.input_size].T.dot(dJdy)
        if optimizer:
            optimizer.update(self, self.dJdW_gradient(dJdy))
        return dJdx

    def dJdW_gradient(self, dJdy):
        dJdW = np.multiply(np.matrix(self.x).T, dJdy).T + self.L1 * np.sign(self.W) + self.L2 * self.W
        return dJdW


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

class SumLayer(GenericLayer):
    def forward(self, x, update = False):
        self.x = np.array(x)
        return np.sum(self.x,0)

    def backward(self, dJdy, optimizer = None):
        return np.ones(self.x.shape)*dJdy

class MulLayer(GenericLayer):
    def forward(self, x, update = False):
        self.x = np.array(x)
        return np.prod(self.x,0)

    def backward(self, dJdy, optimizer = None):
        dJdx = []
        for i in range(self.x.shape[0]):
            dJdx.append(np.prod(np.delete(self.x,i,0),0))
        return np.array(dJdx)*dJdy

class ConstantLayer(GenericLayer):
    def __init__(self, value):
        self.value = value

    def forward(self, x, update = False):
        return self.value

    def backward(self, dJdy, optimizer = None):
        return np.zeros(self.value.size)

class NormalizationLayer(GenericLayer):
    def __init__(self, min_in, max_in, min_out = 0, max_out = 1):
        self.min_in = min_in
        self.max_in = max_in
        self.min_out = min_out
        self.max_out = max_out

    def forward(self, x, update = False):
        return (x/(self.max_in-self.min_in)-self.min_in)*(self.max_out-self.min_out)+self.min_out

    def backward(self, dJdy, optimizer = None):
        return dJdy*(self.max_out-self.min_out)/(self.max_in-self.min_in)