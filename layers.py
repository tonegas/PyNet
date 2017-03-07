import numpy as np
from genericlayer import GenericLayer

# def checkInputDim(self,x):
#     x = np.array(x)
#     if self.num_inputs is not None:
#         if x.shape != (self.num_inputs,):
#             raise Exception('Wrong dimension!')
#     else:
#         self.num_inputs = x.shape[0]


class LinearLayer(GenericLayer):
    def __init__(self, num_inputs, num_outputs, weights = 'random'):
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

    def backward(self, in_delta):
        out_delta = self.W[:,0:self.num_inputs].T.dot(in_delta)
        return out_delta

    def update(self, in_delta):
        dW = np.multiply(np.matrix(self.x).T, in_delta).T
        #dW = (np.identity(self.num_outputs)*in_delta).dot(self.x*np.ones([self.num_outputs,self.num_inputs+1]))
        self.W += dW
        return dW

class SoftMaxLayer(GenericLayer):
    def forward(self, x):
        exp_x = np.exp(x-np.max(x))
        self.y = exp_x/np.sum(exp_x)
        return self.y

    def backward(self, in_delta):
        out_delta = np.zeros(in_delta.size)
        for i in range(self.y.size):
            aux_y = -self.y.copy()
            aux_y[i] = (1-self.y[i])
            out_delta[i] = self.y[i]*aux_y.dot(in_delta)
        return out_delta

class UnitStepLayer(GenericLayer):
    def forward(self, x):
        self.y = 1 if x>= 0 else 0
        return self.y

    def backward(self, in_delta):
        return (-1 if self.y == 0 else 1)*in_delta

class SigmoidLayer(GenericLayer):
    def forward(self, x):
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, in_delta):
        return self.y*(1-self.y)*in_delta

class ReluLayer(GenericLayer):
    def forward(self, x):
        self.x = x
        return np.maximum(0,x)

    def backward(self, in_delta):
        return np.maximum(0,self.x > 0)*in_delta
