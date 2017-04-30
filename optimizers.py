import numpy as np

class Optimizer(object):
    def __init__(self):
        self.store = False

    def update(self, layer, dJdW):
        pass

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super(GradientDescent, self).__init__()
        self.learning_rate = learning_rate

    def update(self, layer, dJdW):
        if self.store:
            layer.dW += dJdW
        else:
            layer.W -= self.learning_rate * (layer.dW + dJdW)
            layer.dW.fill(0.0)

class GradientDescentMomentum(Optimizer):
    def __init__(self, learning_rate, momentum):
        super(GradientDescentMomentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer, dJdW):
        if not hasattr(layer,'velocity'):
            layer.velocity = 0
        layer.velocity = (self.momentum*layer.velocity)-(self.learning_rate*dJdW)

        if self.store:
            layer.dW += layer.velocity
        else:
            layer.W += (layer.dW + layer.velocity)
            layer.dW.fill(0.0)

class AdaGrad(Optimizer):
    def __init__(self, learning_rate):
        super(AdaGrad, self).__init__()
        self.learning_rate = learning_rate
        self.delta = 10**-7

    def update(self, layer, dJdW):
        if not hasattr(layer,'accumulation'):
            layer.r = 0
        layer.r += np.multiply(dJdW,dJdW)

        variation = self.learning_rate/(self.delta+np.sqrt(layer.r))
        if self.store:
            layer.dW -= np.multiply(variation,dJdW)
        else:
            layer.W += layer.dW - np.multiply(variation,dJdW)
            layer.dW.fill(0.0)

class RmsProp():
    pass