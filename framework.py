import numpy as np
from itertools import izip

# def checkInputDim(self,x):
#     x = np.array(x)
#     if self.num_inputs is not None:
#         if x.shape != (self.num_inputs,):
#             raise Exception('Wrong dimension!')
#     else:
#         self.num_inputs = x.shape[0]



class GenericLayer:
    def numeric_gradient(self,x):
        dx = 0.000001
        fx = self.forward(x)
        out_delta = np.zeros([fx.size,x.size])
        for r in xrange(x.size):
            dxvett = np.zeros(x.size)
            dxvett[r] = dx
            fxdx = self.forward(x+dxvett)
            out_delta[:,r] = (fxdx-fx)/dx
        return out_delta

    def forward(self, x):
        return x

    def backward(self, in_delta):
        return in_delta

    def update(self, in_delta):
        return

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

class SquaredLoss(GenericLayer):
    def calc_loss(self, y, t):
        self.J = 0.5*(y-t)**2
        return self.J

    def calc_delta(self, y, t):
        self.in_delta = -(y - t)
        return self.in_delta

class NegativeLogLikehoodLoss(GenericLayer):
    def calc_loss(self, y, t):
        pass

    def calc_delta(self, y, t):
        pass

class SoftMaxNegativeLogLikehoodLoss(GenericLayer):
    def calc_loss(self, y, t):
        pass

    def calc_delta(self, y, t):
        pass

class Sequential:
    def __init__(self, layers = None):
        self.layers = [] if layers is None else layers

    def add(self,layer):
        self.layers.append(layer)

    def forward(self,x):
        aux_x = x
        for layer in self.layers:
            aux_x = layer.forward(aux_x)

        return aux_x

    def backward(self, in_delta, update = False):
        aux_in_delta = in_delta
        for layer in reversed(self.layers):
            aux_out_delta = layer.backward(aux_in_delta)
            if update:
                layer.update(aux_in_delta)

            aux_in_delta = aux_out_delta

        return aux_in_delta

    def learn_one(self, x, t, loss, learning_rate):
        y = self.forward(x)
        J = loss.calc_loss(y,t)
        delta = loss.calc_delta(y,t)
        self.backward(learning_rate * delta, True)
        return J, delta

    def learn(self, x_list, t_list, loss, learning_rate, epochs):
        J_list = np.zeros(x_list.size*epochs)
        delta_list = []
        for epoch in xrange(epochs):
            for x,t in izip(x_list,t_list):
                J, delta = self.learn_one(self, x, t, loss, True)
                J_list.append(J)

        return J, delta
