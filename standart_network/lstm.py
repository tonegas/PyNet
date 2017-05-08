import numpy as np
from genericlayer import GenericLayer
from computationalgraph import Input, MWeight, VWeight, Sigmoid, Tanh, Concat
from layers import ComputationalGraphLayer, VariableDictLayer


# class VanillaNode(GenericLayer):
#     def __init__(self, input_size, output_size,  memory_size, Wxh='gaussian', Whh='gaussian', Why='gaussian', bh='zeros', by='zeros'):
#         self.memory_size = memory_size
#         x = Input(['x','h'],'x')
#         h = Input(['x','h'],'h')
#         s = Input(['s'],'s')
#         self.Wxh = MWeight(input_size, memory_size, weights = Wxh)
#         self.Whh = MWeight(memory_size, memory_size, weights = Whh)
#         self.bh = VWeight(memory_size, weights = bh)
#         self.Why = MWeight(memory_size, output_size, weights = Why)
#         self.by = VWeight(output_size, weights = by)
#         self.statenet = ComputationalGraphLayer(
#                     Tanh(self.Wxh*x+self.Whh*h+self.bh)
#                 )
#         self.outputnet = ComputationalGraphLayer(
#                     self.Why*s+self.by
#                 )
#         self.state = np.zeros(memory_size)
#         self.dJdstate = np.zeros(memory_size)
#
#     def forward(self, x_h, update = False):
#         self.state = self.statenet.forward(x_h)
#         self.y = self.outputnet.forward(self.state)
#         return [self.y,self.state]
#
#     def backward(self, dJdy_dJdh, optimizer = None):
#         dJds = self.outputnet.backward(dJdy_dJdh[0], optimizer)
#         dJdx_dstate = self.statenet.backward(dJds+dJdy_dJdh[1], optimizer)
#         return dJdx_dstate
from network import Sequential


class LSTMNode(GenericLayer):
    def __init__(self, input_size, output_size, Wi='gaussian', Wf='gaussian', Wc='gaussian', Wo='gaussian', bf='zeros', bi='zeros', bc='zeros', bo='zeros'):
        self.input_size = input_size
        self.output_size = output_size
        vars = ['x','h','c']
        x = Input(vars,'x')
        h = Input(vars,'h')
        c = Input(vars,'c')
        Wi = MWeight(input_size+output_size, output_size, weights = Wi)
        Wf = MWeight(input_size+output_size, output_size, weights = Wf)
        Wc = MWeight(input_size+output_size, output_size, weights = Wc)
        bf = VWeight(output_size, weights = bf)
        bi = VWeight(output_size, weights = bi)
        bc = VWeight(output_size, weights = bc)
        Wo = MWeight(input_size+output_size, output_size, weights = Wo)
        bo = VWeight(output_size, weights = bo)
        self.ct_net = Sequential(
            VariableDictLayer(vars),
            ComputationalGraphLayer(
                Sigmoid(Wf.dot(Concat([x,h]))+bf)*c+
                Sigmoid(Wi.dot(Concat([x,h]))+bi)*Tanh(Wc.dot(Concat([x,h]))+bc)
            )
        )
        print self.ct_net

        # self.ht_net = ComputationalGraphLayer(
        #     Tanh(c)*(Wo*xh+bo)
        # )
        # self.ct = np.zeros(output_size)
        # self.ht = np.zeros(output_size)
        # self.state = [self.ct,self.ht]
        # self.dJdstate = [np.zeros(output_size),np.zeros(output_size)]

    def forward(self, x, update = False):
        return self.ct_net.forward(x)
        # xh = np.hstack([x_h[0],x_h[1][0]])
        # self.ct = self.ct_net.forward([xh,x_h[1][1]])
        # self.ht = self.ht_net.forward([xh,self.ct])
        # self.state = [self.ct,self.ht]
        # return [self.ht,self.state]

    def backward(self, dJdy_dJdh, optimizer = None):
        pass
        # dJdxh_dJdc = self.ht_net.backward(dJdy_dJdh[0]+dJdy_dJdh[1][1], optimizer)
        # dJdxh_dJdc = self.ct_net.backward(dJdy_dJdh[1][0]+dJdxh_dJdc[1], optimizer)
        #
        #
        # dJdx_dstate = self.statenet.backward(dJds+dJdy_dJdh[1], optimizer)
        # return dJdx_dstate

    def backward(self, dJdy,  optimizer = None):
        pass
        # dJdx_group = self.ht_net.backward(dJdy, optimizer)
        #
        # dJdx = self.ct_net.backward(dJdx_group[0], optimizer)
        # return dJdx
#
lstm = LSTMNode(2,3)
inputx = {'x':np.array([2,3]),'h':np.array([1,2,3]),'c':np.array([3,2,1])}
print lstm.forward(inputx)
