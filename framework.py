import numpy as np

class LinearLayer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.W = np.random.rand(outputs, inputs+1)

    def forward(self, x):
        ones = np.ones(self.outputs)
        x = np.hstack([ones, x])
        self.x = x
        y = self.W.dot(x)
        return y

    def backward(self, dJdy):
        return self.W.dot(dJdy)

    def update(self, dJdy):
        self.W += self.x.multiply(dJdy)

class SquaredErrorLayer:

    def backward(self):
        return alpha

model = Sequential()
model.add(LinearLayer(2, 1))
model.add(SigmoidLayer())
