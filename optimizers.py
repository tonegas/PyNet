import numpy as np

class Optimizer(object):
    def __init__(self, clip=None):
        self.weight_list = {}
        self.clip = clip

    def update_dW(self, weight, dJdW):
        weight.dW += dJdW
        self.weight_list[weight.W.ctypes.data] = weight

    def update_W(self, layer):
        pass

    def update_model(self):
        # print self.weight_list
        for weight in self.weight_list:
            if self.clip is not None:
                np.clip(self.weight_list[weight].dW, -self.clip, self.clip, out=self.weight_list[weight].dW)
            self.update_W(self.weight_list[weight])

class GradientDescent(Optimizer):
    def __init__(self, learning_rate, **kwargs):
        super(GradientDescent, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def update_W(self, weight):
        weight.W -= self.learning_rate * weight.dW
        weight.dW.fill(0.0)

class GradientDescentMomentum(Optimizer):
    def __init__(self, learning_rate, momentum, **kwargs):
        super(GradientDescentMomentum, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_W(self, weight):
        if not hasattr(weight,'velocity'):
            weight.velocity = 0
        weight.velocity = (self.momentum*weight.velocity)-(self.learning_rate*weight.dW)

        weight.W += weight.velocity
        weight.dW.fill(0.0)

class AdaGrad(Optimizer):
    def __init__(self, learning_rate, **kwargs):
        super(AdaGrad, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.delta = 1e-8

    def update_W(self, weight):
        if not hasattr(weight, 'memory'):
            weight.memory = 0
        weight.memory += np.multiply(weight.dW, weight.dW)

        weight.W += -(self.learning_rate * weight.dW) / np.sqrt(weight.memory + self.delta)
        weight.dW.fill(0.0)

class RmsProp():
    pass