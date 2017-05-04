import numpy as np
import collections

from genericlayer import GenericLayer
from utils import SharedWeights
from network import Sequential
from layers import  TanhLayer, LinearLayer, MWeightLayer, ComputationalGraphLayer
from computationalgraph import MWeight, VWeight, Input, Tanh, Softmax
from groupnetworks import SumGroup

class Vanilla(GenericLayer):
    def __init__(self, input_size, output_size,  memory_size, window_size, Wxh='gaussian', Whh='gaussian', Why='gaussian', bh='zeros', by='zeros'):
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

        self.Wxh = SharedWeights(Wxh)
        self.Whh = SharedWeights(Whh)
        self.Why = SharedWeights(Why)
        self.bh = SharedWeights(bh)
        self.by = SharedWeights(by)

        for ind in range(window_size):
            cWxh = MWeight(input_size, memory_size, weights=self.Wxh)
            cWhh = MWeight(memory_size, memory_size, weights=self.Whh)
            cbh = VWeight(memory_size, weights=self.bh)
            cWhy = MWeight(memory_size, output_size, weights=self.Why)
            cby = VWeight(output_size, weights=self.by)
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
        # print 'state'+str(self.state[self.window_step])
        y = self.outputnet[self.window_step].forward(self.state[self.window_step])
        # self.dJdh[self.window_step] = np.zeros(memory_size)
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
