import numpy as np

class Optimizer(object):
    def __init__(self, clip=None):
        self.weight_list = {}
        self.weight_params = {}
        self.clip = clip

    def update_dW(self, weight, dJdW):
        weight.dW += dJdW + weight.L1 * np.sign(weight.get()) + weight.L2 * weight.get()
        self.weight_list[weight.W.ctypes.data] = weight

    def update_W(self, weight):
        pass

    def get_or_create_param(self, weight, param_id, param_init_val = 0.0):
        if weight.W.ctypes.data not in  self.weight_params:
            self.weight_params[weight.W.ctypes.data] = {}
        return self.weight_params[weight.W.ctypes.data].get(param_id, param_init_val)

    def set_param(self, weight, param_id, param_val):
        self.weight_params[weight.W.ctypes.data][param_id] = param_val

    def update_model(self):
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
        velocity = (self.momentum*self.get_or_create_param(weight,'velocity')) - (self.learning_rate*weight.dW)
        self.set_param(weight, 'velocity', velocity)
        weight.W += velocity
        weight.dW.fill(0.0)

class AdaGrad(Optimizer):
    def __init__(self, learning_rate, **kwargs):
        super(AdaGrad, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.delta = 1e-8

    def update_W(self, weight):
        r = self.get_or_create_param(weight, 'r') + np.multiply(weight.dW, weight.dW)
        self.set_param(weight, 'r', r)

        weight.W += -(self.learning_rate * weight.dW) / np.sqrt(r + self.delta)
        weight.dW.fill(0.0)

class RmsProp():
    pass