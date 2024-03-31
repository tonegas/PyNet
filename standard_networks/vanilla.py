import numpy as np
import collections

from genericlayer import GenericLayer
from utils import SharedWeights
from network import Sequential
from layers import  TanhLayer, LinearLayer, MWeightLayer, ComputationalGraphLayer
from computationalgraph import MWeight, VWeight, Input, Tanh, Softmax
from recursivenetwork import RNN
from groupnetworks import SumGroup

class VanillaNet(RNN):
    def __init__(self, input_size, output_size,  memory_size, Wxh='gaussian', Whh='gaussian', Why='gaussian', bh='zeros', by='zeros'):
        self.Wxh = SharedWeights.get_or_create(Wxh, input_size, memory_size)
        self.Whh = SharedWeights.get_or_create(Whh, memory_size, memory_size)
        self.Why = SharedWeights.get_or_create(Why, memory_size, output_size)
        self.bh = SharedWeights.get_or_create(bh, 1, memory_size)
        self.by = SharedWeights.get_or_create(by, 1, output_size)
        RNN.__init__(self, VanillaNode, input_size, output_size,  memory_size, Wxh=self.Wxh, Whh=self.Whh, Why=self.Why, bh=self.bh, by=self.by)


class VanillaNode(GenericLayer):
    def __init__(self, input_size, output_size,  memory_size, Wxh='gaussian', Whh='gaussian', Why='gaussian', bh='zeros', by='zeros'):
        self.memory_size = memory_size
        x = Input(['x','h'],'x')
        h = Input(['x','h'],'h')
        s = Input(['s'],'s')
        self.Wxh = MWeight(input_size, memory_size, weights = Wxh)
        self.Whh = MWeight(memory_size, memory_size, weights = Whh)
        self.bh = VWeight(memory_size, weights = bh)
        self.Why = MWeight(memory_size, output_size, weights = Why)
        self.by = VWeight(output_size, weights = by)
        self.statenet = ComputationalGraphLayer(
                    Tanh(self.Wxh.dot(x)+self.Whh.dot(h)+self.bh)
                )
        self.outputnet = ComputationalGraphLayer(
                    self.Why.dot(s)+self.by
                )
        self.state = np.zeros(memory_size)
        self.dJdstate = np.zeros(memory_size)

    def forward(self, x_h, update = False):
        self.state = self.statenet.forward(x_h)
        self.y = self.outputnet.forward(self.state)
        return [self.y,self.state]

    def backward(self, dJdy_dJdh, optimizer = None):
        dJds = self.outputnet.backward(dJdy_dJdh[0], optimizer)
        dJdx_dstate = self.statenet.backward(dJds+dJdy_dJdh[1], optimizer)
        return dJdx_dstate

class Vanilla(GenericLayer):
    def __init__(self, input_size, output_size, memory_size, window_size, Wxh='gaussian', Whh='gaussian', Why='gaussian', bh='zeros', by='zeros'):
        self.memory_size = memory_size
        self.window_size = window_size
        self.window_step = 0
        x = Input(['x','h'],'x')
        h = Input(['x','h'],'h')
        s = Input(['s'],'s')
        self.statenet = []
        self.outputnet = []
        self.state = []
        self.dJdh = []

        self.Wxh = SharedWeights(Wxh, input_size, memory_size)
        self.Whh = SharedWeights(Whh, memory_size, memory_size)
        self.Why = SharedWeights(Why, memory_size, output_size)
        self.bh = SharedWeights(bh, 1, memory_size)
        self.by = SharedWeights(by, 1, output_size)

        for ind in range(window_size):
            cWxh = MWeight(input_size, memory_size, weights = self.Wxh)
            cWhh = MWeight(memory_size, memory_size, weights = self.Whh)
            cbh = VWeight(memory_size, weights = self.bh)
            cWhy = MWeight(memory_size, output_size, weights = self.Why)
            cby = VWeight(output_size, weights = self.by)
            self.statenet.append(
                ComputationalGraphLayer(
                    Tanh(cWxh*x+cWhh*h+cbh)
                )
            )
            self.outputnet.append(
                ComputationalGraphLayer(
                    # Softmax(cWhy*s+cby)
                    cWhy*s+cby
                )
            )
            self.state.append(np.zeros(memory_size))
            self.dJdh.append(np.zeros(memory_size))

    def clear_memory(self):
        for ind in range(self.window_size):
            self.state[ind].fill(0.0)

    def forward(self, x, update = False):
        self.state[self.window_step] = self.statenet[self.window_step].forward([x,self.state[self.window_step-1]])
        y = self.outputnet[self.window_step].forward(self.state[self.window_step])
        self.window_step += 1
        if self.window_step >= self.window_size:
            self.window_step = 0
        return y

    def backward(self, dJdy, optimizer = None):
        # print 'back'
        self.window_step -= 1
        if self.window_step < 0:
            self.window_step = self.window_size-1
        # print dJdy
        dJdx = self.outputnet[self.window_step].backward(dJdy, optimizer)
        dJdx_dJdh = self.statenet[self.window_step].backward(dJdx+self.dJdh[self.window_step], optimizer)
        if self.window_step > 0:
            self.dJdh[self.window_step-1] = dJdx_dJdh[1]
        return dJdx_dJdh[0]
