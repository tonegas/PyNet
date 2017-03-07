import numpy as np
from genericlayer import GenericLayer

class SquaredLoss(GenericLayer):
    def forward(self, x):
        return self.calc_loss(x, self.t)

    def backward(self, in_delta):
        return -self.calc_gradient(in_delta, self.t)

    def calc_loss(self, y, t):
        self.t = t
        self.J = 0.5*(t-y)**2
        return self.J

    def calc_gradient(self, y, t):
        #the gradient is positive but delta is negative
        self.grad = (y - t)
        return self.grad

class NegativeLogLikelihoodLoss(GenericLayer):
    def forward(self, x):
        return self.calc_loss(x, self.t)

    def backward(self, in_delta):
        return self.calc_gradient(in_delta, self.t)

    def calc_loss(self, y, t):
        self.t = t
        return -t*np.log(y)

    def calc_gradient(self, y, t):
        #the gradient is positive but delta is negative
        return -t/y

class CrossEntropyLoss(GenericLayer):
    def forward(self, x):
        return self.calc_loss(x, self.t)

    def backward(self, in_delta):
        return self.calc_gradient(in_delta, self.t)

    def calc_loss(self, y, t):
        self.t = t
        totlog = np.log(np.sum(np.exp(y)))
        return t*(totlog - y)

    def calc_gradient(self, y, t):
        exp_y = np.exp(y-np.max(y))
        self.y = exp_y/np.sum(exp_y)
        return t-self.y