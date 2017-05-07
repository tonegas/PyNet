import numpy as np
from genericlayer import GenericLayer


class SquaredLoss(GenericLayer):
    def forward(self, x, update = False):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        return 0.5*(t-y)**2

    def dJdy_gradient(self, y, t):
        return (y - t)

#Binary classification (one output [0,1])
#negative log likelihood of the Bernoulli distribution
#J = t*log(y)-(1-t)*log(1-y)

#multiclass cross-entropy (n output, sum(y) = 1)
#negative log likelihood of the multinomial distribution
class NegativeLogLikelihoodLoss(GenericLayer):
    def forward(self, x, update = False):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        return -t*np.log(np.maximum(y,0.0000001))
        # return -t*np.log(y)

    def dJdy_gradient(self, y, t):
        return -t/(np.maximum(y,0.0000001))

#multiclass cross-entropy (n output, sum(y) = 1)
#Negative loglikelihood with softmax output
class CrossEntropyLoss(GenericLayer):
    def forward(self, x, update = False):
        return self.loss(x, self.t)

    #Here max y is neglected
    def loss(self, y, t):
        self.t = t
        totlog = np.log(np.sum(np.exp(y)))
        return t*(totlog - y)

    def dJdy_gradient(self, y, t):
        exp_y = np.exp(y-np.max(y))
        return (exp_y/np.sum(exp_y))-t