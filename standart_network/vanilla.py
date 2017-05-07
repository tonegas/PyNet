import numpy as np
import collections

from genericlayer import GenericLayer
from utils import SharedWeights
from network import Sequential
from layers import  TanhLayer, LinearLayer, MWeightLayer, ComputationalGraphLayer
from computationalgraph import MWeight, VWeight, Input, Tanh, Softmax
from groupnetworks import SumGroup

class NodeGenerator():
    def __init__(self, node, *args, **kwargs):
        self.node = node
        self.args = args
        self.kwargs = kwargs

class RNN(GenericLayer, NodeGenerator):
    def __init__(self, Node, *args, **kwargs):
        NodeGenerator.__init__(self, Node, *args, **kwargs)
        self.window_size = 0
        self.window_step = 0
        self.nodes = []
        self.net = self.node(*self.args, **self.kwargs)
        self.message_fun = {
            'delete_nodes' : self.delete_nodes,
            'init_nodes' : self.init_nodes,
            'clear_memory' : self.clear_memory
        }

    def on_message(self,message,*args,**kwargs):
        self.message_fun[message](*args,**kwargs)

    def delete_nodes(self):
        self.nodes = []

    def init_nodes(self, window_size):
        self.window_size = window_size
        self.window_step = 0
        for ind in range(window_size):
            #self.nodes[ind].robba()
            self.nodes.append(self.node(*self.args, **self.kwargs))

    def clear_memory(self):
        self.net.state.fill(0.0)
        for ind in range(self.window_size):
            self.nodes[ind].state.fill(0.0)
            self.nodes[ind].dJdstate.fill(0.0)


    def forward(self, x, update = False):
        if update:
            [y, self.nodes[self.window_step].state] = self.nodes[self.window_step].forward([x, self.nodes[self.window_step - 1].state])
            self.window_step += 1
            if self.window_step >= self.window_size:
                self.window_step = 0
            return y
        else:
            [y,self.net.state] = self.net.forward([x, self.net.state])
            return y

    def backward(self, dJdy, optimizer = None):
        # print 'back'
        self.window_step -= 1
        if self.window_step < 0:
            self.window_step = self.window_size-1
        [dJdx, dJdh] = self.nodes[self.window_step].backward([dJdy, self.nodes[self.window_step].dJdstate], optimizer)
        if self.window_step > 0:
            self.nodes[self.window_step-1].dJdstate = dJdh
        return dJdx


class VanillaNet(RNN):
    def __init__(self, input_size, output_size,  memory_size, Wxh='gaussian', Whh='gaussian', Why='gaussian', bh='zeros', by='zeros'):
        self.Wxh = SharedWeights(Wxh, input_size, memory_size)
        self.Whh = SharedWeights(Whh, memory_size, memory_size)
        self.Why = SharedWeights(Why, memory_size, output_size)
        self.bh = SharedWeights(bh, 1, memory_size)
        self.by = SharedWeights(by, 1, output_size)
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
                    Tanh(self.Wxh*x+self.Whh*h+self.bh)
                )
        self.outputnet = ComputationalGraphLayer(
                    self.Why*s+self.by
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
