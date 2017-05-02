import numpy as np

class Optimizer(object):
    def __init__(self):
        self.layer_list = {}

    def update_dW(self, layer, dJdW):
        layer.dW += dJdW
        self.layer_list[layer.W.ctypes.data] = layer

    def update_W(self, layer):
        pass

    def update_model(self):
        for layer in self.layer_list:
            self.update_W(self.layer_list[layer])

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super(GradientDescent, self).__init__()
        self.learning_rate = learning_rate

    def update_W(self, layer):
        layer.W -= self.learning_rate * layer.dW
        layer.dW.fill(0.0)

class GradientDescentMomentum(Optimizer):
    def __init__(self, learning_rate, momentum):
        super(GradientDescentMomentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_W(self, layer):
        if not hasattr(layer,'velocity'):
            layer.velocity = 0
        layer.velocity = (self.momentum*layer.velocity)-(self.learning_rate*layer.dW)

        layer.W += layer.velocity
        layer.dW.fill(0.0)

class AdaGrad(Optimizer):
    def __init__(self, learning_rate):
        super(AdaGrad, self).__init__()
        self.learning_rate = learning_rate
        self.delta = 1e-8

    def update_W(self, layer):
        if not hasattr(layer,'memory'):
            layer.memory = 0
        layer.memory += np.multiply(layer.dW,layer.dW)

        layer.W += -(self.learning_rate*layer.dW)/np.sqrt(layer.memory+self.delta)
        layer.dW.fill(0.0)

class RmsProp():
    pass