import numpy as np


class GenericLayer:
    def __init__(self):
        self.group = False

    def numeric_gradient(self,x):
        dx = 0.00000001
        fx = self.forward(x)
        dJdx = np.zeros([fx.size,x.size])
        for r in xrange(x.size):
            dxvett = np.zeros(x.size)
            dxvett[r] = dx
            fxdx = self.forward(x+dxvett)
            dJdx[:,r] = (fxdx-fx)/dx
        return dJdx

    def forward(self, x):
        return x

    def backward(self, dJdy):
        return dJdy