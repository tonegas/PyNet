import numpy as np
from genericlayer import GenericLayer
from network import Sequential
from layers import  TanhLayer, LinearLayer, MWeightLayer
from groupnetworks import SumGroup

class Vanilla(GenericLayer):
    def __init__(self, input_size, output_size,  memory_size):
        self.n1 = Sequential(
            SumGroup(
                LinearLayer(input_size, memory_size),
                LinearLayer(memory_size,memory_size)
            ),
            TanhLayer,
        )
        self.n2 = MWeightLayer(memory_size, output_size)
        self.h = np.zeros(memory_size)

    def forward(self, x, update = False):
        self.h = self.n1.forward([x,self.h])
        return self.n2.forward(self.h)

    def backward(self, dJdy, optimizer = None):
        dJdx = self.n2.backward(dJdy, optimizer)
        dJdx_dJdh = self.n1.backward(dJdx, optimizer)
        return dJdx_dJdh[0]
