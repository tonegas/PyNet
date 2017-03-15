import numpy as np
from genericlayer import GenericLayer


class LinearLayer(GenericLayer):
    def __init__(self, num_inputs, num_outputs, weights = 'random', L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if type(weights) == str:
            if weights == 'random':
                self.W = np.random.rand(num_outputs, num_inputs+1)
            elif weights == 'ones':
                self.W = np.ones([num_outputs, num_inputs+1])
        elif type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
            self.W = weights
        else:
            raise Exception('Type not correct!')

    def forward(self, x):
        self.x = np.hstack([x, 1])
        return self.W.dot(self.x)

    def backward(self, dJdy):
        dJdx = self.W[:,0:self.num_inputs].T.dot(dJdy)
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

class UnitStepLayer(GenericLayer):
    def forward(self, x):
        self.y = (x >= 0)*1.0
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
        return np.maximum(-1,self.x > 0)*dJdy
