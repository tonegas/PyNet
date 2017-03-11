import numpy as np
from genericlayer import GenericLayer

class SquaredLoss(GenericLayer):
    def forward(self, x):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        return 0.5*(t-y)**2

    def dJdy_gradient(self, y, t):
        return (y - t)

class NegativeLogLikelihoodLoss(GenericLayer):
    def forward(self, x):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        return -t*np.log(y)

    def dJdy_gradient(self, y, t):
        return -t/y

class CrossEntropyLoss(GenericLayer):
    def forward(self, x):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        totlog = np.log(np.sum(np.exp(y)))
        return t*(totlog - y)

    def dJdy_gradient(self, y, t):
        exp_y = np.exp(y-np.max(y))
        return (exp_y/np.sum(exp_y))-t