import numpy as np
from genericlayer import GenericLayer

def to_one_hot_vect(vect, num_classes):
    on_hot_vect = []
    for i,target in enumerate(vect):
        on_hot_vect.append(np.zeros(num_classes))
        on_hot_vect[i][target] = 1
    return on_hot_vect

class SquaredLoss(GenericLayer):
    def forward(self, x, update = False):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        return 0.5*(t-y)**2

    def dJdy_gradient(self, y, t):
        return (y - t)

class NegativeLogLikelihoodLoss(GenericLayer):
    def forward(self, x, update = False):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        return -t*np.log(np.maximum(y,0.0000001))
        # return -t*np.log(y)

    def dJdy_gradient(self, y, t):
        return -t/(np.maximum(y,0.000001))

class CrossEntropyLoss(GenericLayer):
    def forward(self, x, update = False):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        totlog = np.log(np.sum(np.exp(y)))
        return t*(totlog - y)

    def dJdy_gradient(self, y, t):
        exp_y = np.exp(y-np.max(y))
        return (exp_y/np.sum(exp_y))-t