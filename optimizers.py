import numpy as np

class Optimizer():
    def __init__(self):
        self.store = False

    def update(self, layer, dJdW):
        pass

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.lerning_rate = learning_rate

    def update(self, layer, dJdW):
        if self.store:
            layer.dW += dJdW
        else:
            layer.W -= self.lerning_rate * (layer.dW + dJdW)
            layer.dW = 0

class GradientDescentMomentum(Optimizer):
    def __init__(self, learning_rate, momentum):
        self.lerning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer, dJdW):
        if not hasattr(layer,'velocity'):
            layer.velocity = 0
        else:
            layer.velocity = (self.momentum*layer.velocity)-(self.lerning_rate*dJdW)

        if self.store:
            layer.dW += layer.velocity
        else:
            layer.W += (layer.dW + layer.velocity)
            layer.dW = 0

        # layer.W += layer.velocity

class AdaGrad():
    pass

class RmsProp():
    pass