import numpy as np
from genericlayer import GenericLayer
from network import Sequential
from layers import  TanhLayer, LinearLayer, MWeightLayer, ComputationalGraphLayer
from computationalgraph import MWeight, VWeight, Input, Tanh, Softmax
from groupnetworks import SumGroup

class Vanilla(GenericLayer):
    def __init__(self, input_size, output_size,  memory_size):
        Wx = MWeight(input_size,memory_size,weights='gaussian')
        Wh = MWeight(memory_size,memory_size,weights='gaussian')
        b = VWeight(memory_size,weights='zeros')
        Wo = MWeight(memory_size,output_size,weights='gaussian')
        o = VWeight(output_size,weights='zeros')
        x = Input(['x','h'],'x')
        h = Input(['x','h'],'h')
        self.statenet = ComputationalGraphLayer(
            Tanh(Wx*x+Wh*h+b)
        )
        s = Input(['s'],'s')
        self.outputnet = ComputationalGraphLayer(
            Softmax(Wo*s+o)
        )
        self.state = np.zeros(memory_size)
        self.dJdh = np.zeros(memory_size)

    def forward(self, x, update = False):
        self.state = self.statenet.forward([x,self.state])
        return self.outputnet.forward(self.state)

    def backward(self, dJdy, optimizer = None):
        dJdx = self.outputnet.backward(dJdy, optimizer)
        dJdx_dJdh = self.statenet.backward(dJdx+self.dJdh, optimizer)
        #self.dJdh = (dJdx_dJdh[1])/np.max(dJdx_dJdh[1])
        self.dJdh = dJdx_dJdh[1]
        return dJdx_dJdh[0]
