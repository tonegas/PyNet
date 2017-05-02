import numpy as np
import collections

from genericlayer import GenericLayer
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

        self.dWxh = np.zeros_like(Wxh)
        self.dWhh = np.zeros_like(Whh)
        self.dWhy = np.zeros_like(Why)
        self.dbh = np.zeros_like(bh)
        self.dby = np.zeros_like(by)
        for ind in range(window_size):
            cWxh = MWeight(input_size, memory_size, weights=Wxh, dweights=self.dWxh)
            cWhh = MWeight(memory_size, memory_size, weights=Whh, dweights=self.dWhh)
            cbh = VWeight(memory_size, weights=bh, dweights=self.dbh)
            cWhy = MWeight(memory_size, output_size, weights=Why, dweights=self.dWhy)
            cby = VWeight(output_size, weights=by, dweights=self.dby)
            self.statenet.append(
                ComputationalGraphLayer(
                    Tanh(cWxh*x+cWhh*h+cbh)
                )
            )
            self.outputnet.append(
                ComputationalGraphLayer(
                    Softmax(cWhy*s+cby)
                )
            )
            self.state.append(np.zeros(memory_size))
            self.dJdh.append(np.zeros(memory_size))

    def forward(self, x, update = False):
        if self.window_step > 0:
            self.state[self.window_step] = self.statenet[self.window_step].forward([x,self.state[self.window_step-1]])
        else:
            self.state[self.window_step] = self.statenet[self.window_step].forward([x,np.zeros(self.memory_size)])
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
