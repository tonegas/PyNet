import numpy as np


class GenericLayer:
    def numeric_gradient(self,x):
        dx = 0.0000001
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