import numpy as np

class StocaticGradientDescent():
    def __init__(self, learning_rate):
        self.lerning_rate = learning_rate

    def update(self, layer, dJdW):
        layer.W -= self.lerning_rate * dJdW


class SGDMomentum():
    def __init__(self, learning_rate, momentum):
        self.lerning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer, dJdW):
        if not hasattr(layer,'velocity'):
            layer.velocity = 0
        else:
            layer.velocity = (self.momentum*layer.velocity)-(self.lerning_rate*dJdW)
        layer.W += layer.velocity

class AdaGrad():
    pass

class RmsProp():
    pass