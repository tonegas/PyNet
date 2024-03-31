import numpy as np
from genericlayer import GenericLayer
from computationalgraph import Input, MWeight, VWeight, Sigmoid, Tanh, Concat
from layers import ComputationalGraphLayer, VariableDictLayer
from network import Sequential
from recursivenetwork import RNN
from utils import SharedWeights

class LSTMNet(RNN):
    def __init__(self, input_size, output_size, Wi='gaussian', Wf='gaussian', Wc='gaussian', Wo='gaussian', bi='zeros', bf='zeros', bc='zeros', bo='zeros'):
        self.Wi = SharedWeights.get_or_create(Wi, input_size+output_size, output_size)
        self.Wf = SharedWeights.get_or_create(Wf, input_size+output_size, output_size)
        self.Wc = SharedWeights.get_or_create(Wc, input_size+output_size, output_size)
        self.Wo = SharedWeights.get_or_create(Wo, input_size+output_size, output_size)
        self.bi = SharedWeights.get_or_create(bi, 1, output_size)
        self.bf = SharedWeights.get_or_create(bf, 1, output_size)
        self.bc = SharedWeights.get_or_create(bc, 1, output_size)
        self.bo = SharedWeights.get_or_create(bo, 1, output_size)
        RNN.__init__(self, LSTMNode, input_size, output_size,  Wi=self.Wi, Wf=self.Wf, Wc=self.Wc, Wo=self.Wo, bi=self.bi, bf=self.bf, bc=self.bc, bo=self.bo)

class LSTMNode(GenericLayer):
    def __init__(self, input_size, output_size, Wi='gaussian', Wf='gaussian', Wc='gaussian', Wo='gaussian', bi='zeros', bf='zeros', bc='zeros', bo='zeros'):
        self.input_size = input_size
        self.output_size = output_size
        vars = ['x','h','c']
        x = Input(vars,'x')
        h = Input(vars,'h')
        c = Input(vars,'c')
        Wi = MWeight(input_size+output_size, output_size, weights = Wi)
        Wf = MWeight(input_size+output_size, output_size, weights = Wf)
        Wc = MWeight(input_size+output_size, output_size, weights = Wc)
        Wo = MWeight(input_size+output_size, output_size, weights = Wo)
        bi = VWeight(output_size, weights = bi)
        bf = VWeight(output_size, weights = bf)
        bc = VWeight(output_size, weights = bc)
        bo = VWeight(output_size, weights = bo)
        self.ct_net = Sequential(
            VariableDictLayer(vars),
            ComputationalGraphLayer(
                Sigmoid(Wf.dot(Concat([x,h]))+bf)*c+
                Sigmoid(Wi.dot(Concat([x,h]))+bi)*Tanh(Wc.dot(Concat([x,h]))+bc)
            )
        )
        self.ht_net = Sequential(
            VariableDictLayer(vars),
            ComputationalGraphLayer(
                Tanh(c)*(Wo.dot(Concat([x,h]))+bo)
            )
        )
        self.ct = np.zeros(output_size)
        self.ht = np.zeros(output_size)
        self.state = [self.ct,self.ht]
        self.dJdstate = [np.zeros(output_size),np.zeros(output_size)]

    def forward(self, x_state, update = False):
        xhc = {'x':x_state[0],'h':x_state[1][0],'c':x_state[1][1]}
        self.ct = self.ct_net.forward(xhc)
        xhc['c'] = self.ct
        self.ht = self.ht_net.forward(xhc)
        self.state = [self.ht,self.ct]
        return [self.ht, self.state]

    def backward(self, dJdy_dJdstate, optimizer = None):
        # print dJdy_dJdstate
        dJdydhdc = {'y':dJdy_dJdstate[0],'h':dJdy_dJdstate[1][0],'c':dJdy_dJdstate[1][1]}
        dJdxdhdc = self.ht_net.backward(dJdydhdc['y']+dJdydhdc['h'], optimizer)
        dJct_net = self.ct_net.backward(dJdxdhdc['c']+dJdydhdc['c'], optimizer)
        dJdxdhdc['c'] = dJct_net['c']
        dJdxdhdc['h'] += dJct_net['h']
        dJdxdhdc['x'] += dJct_net['x']
        dJdx_dstate = [dJdxdhdc['x'], [dJdxdhdc['h'],dJdxdhdc['c']]]
        return dJdx_dstate

