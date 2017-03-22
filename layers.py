import numpy as np
from genericlayer import GenericLayer


class LinearLayer(GenericLayer):
    def __init__(self, input_size, output_size, weights ='random', L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        self.input_size = input_size
        self.output_size = output_size
        if type(weights) == str:
            if weights == 'random':
                self.W = np.random.rand(output_size, input_size + 1)
            elif weights == 'ones':
                self.W = np.ones([output_size, input_size + 1])
            elif weights == 'zeros':
                self.W = np.zeros([output_size, input_size + 1])
        elif type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
            self.W = weights
        else:
            raise Exception('Type not correct!')

    def forward(self, x):
        self.x = np.hstack([x, 1])
        return self.W.dot(self.x)

    def backward(self, dJdy):
        dJdx = self.W[:, 0:self.input_size].T.dot(dJdy)
        return dJdx

    def backward_and_update(self, dJdy, optimizer, depth):
        dJdx = self.backward(dJdy)
        optimizer.update(self, self.dJdW_gradient(dJdy))
        return dJdx

    def dJdW_gradient(self, dJdy):
        dJdW = np.multiply(np.matrix(self.x).T, dJdy).T + self.L1 * np.sign(self.W) + self.L2 * self.W
        return dJdW


class SoftMaxLayer(GenericLayer):
    def forward(self, x):
        # print 'xS'+str(x)
        exp_x = np.exp(x-np.max(x))
        # print 'exp_x'+str(exp_x)
        self.y = exp_x/np.sum(exp_x)
        return self.y

    def backward(self, dJdy):
        dJdx = np.zeros(dJdy.size)
        for i in range(self.y.size):
            aux_y = -self.y.copy()
            aux_y[i] = (1-self.y[i])
            dJdx[i] = self.y[i]*aux_y.dot(dJdy)
        return dJdx

class HeavisideLayer(GenericLayer):
    def forward(self, x):
        self.y = (x >= 0)*1.0
        return self.y

    def backward(self, dJdy):
        return ((self.y > 0)*1.0+(self.y <= 0)*-1.0)*dJdy

class SignLayer(GenericLayer):
    def forward(self, x):
        self.y = np.sign(x)
        return self.y

    def backward(self, dJdy):
        return ((self.y > 0)*1.0+(self.y <= 0)*-1.0)*dJdy

class TanhLayer(GenericLayer):
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dJdy):
        return (1.-self.y ** 2) * dJdy

class SigmoidLayer(GenericLayer):
    def forward(self, x):
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, dJdy):
        return self.y*(1-self.y)*dJdy

class ReluLayer(GenericLayer):
    def forward(self, x):
        self.x = x
        return np.maximum(0,x)

    def backward(self, dJdy):
        return np.maximum(0,self.x > 0)*dJdy

class SumLayer(GenericLayer):
    def forward(self, x):
        self.x = np.array(x)
        return np.sum(self.x,0)

    def backward(self, dJdy):
        return np.ones(self.x.shape)*dJdy

class MulLayer(GenericLayer):
    def forward(self, x):
        self.x = np.array(x)
        return np.prod(self.x,0)

    def backward(self, dJdy):
        dJdx = []
        for i in range(self.x.shape[0]):
            dJdx.append(np.prod(np.delete(self.x,i,0),0))
        return np.array(dJdx)*dJdy

class ConstantLayer(GenericLayer):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return self.value

    def backward(self, dJdy):
        return np.zeros(self.value.size)