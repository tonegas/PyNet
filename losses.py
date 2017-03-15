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
        # print t
        # print 'y'+str(y)
        #ly = np.log(y)
        #np.putmask(ly,ly<-10.0**30,-10.0**30)
        # print 'log'+str(np.maximum(np.log(y),-10.0**30))
        return -t*np.log(np.maximum(y,0.0000001))
        # return -t*np.log(y)

    def dJdy_gradient(self, y, t):
        return -t/(np.maximum(y,0.000001))

class CrossEntropyLoss(GenericLayer):
    def forward(self, x):
        return self.loss(x, self.t)

    def loss(self, y, t):
        self.t = t
        # print y[0]
        #y = y/np.sum(y)
        totlog = np.log(np.sum(np.exp(y)))
        return t*(totlog - y)

    def dJdy_gradient(self, y, t):
        exp_y = np.exp(y-np.max(y))
        return (exp_y/np.sum(exp_y))-t