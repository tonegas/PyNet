import numpy as np

from genericlayer import GenericLayer
from layers import define_weights, LinearLayer, SignLayer, SigmoidLayer, TanhLayer, SumLayer, MulLayer, ComputationalGraphLayer
from network import Sequential, SumGroup, MulGroup, ParallelGroup
from computationalgraph import Sigmoid, Weight, MatrixWeight, Input, Tanh


class LSTM(GenericLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.n1 = SumGroup(
            MulGroup(Sequential(LinearLayer(input_size+output_size,output_size),SigmoidLayer),GenericLayer),
            MulGroup(Sequential(LinearLayer(input_size+output_size,output_size),SigmoidLayer),Sequential(LinearLayer(input_size+output_size,output_size),TanhLayer))
        )
        self.n2 = MulGroup(
            Sequential(GenericLayer, TanhLayer),
            Sequential(LinearLayer(input_size+output_size, output_size),SigmoidLayer)
        )

        self.ct = np.zeros(output_size)
        self.ht = np.zeros(output_size)

    def forward(self, x, update = False):
        self.ct = self.n1.forward([[np.append(x,self.ht),self.ct],[np.append(x,self.ht),np.append(x,self.ht)]])
        self.ht = self.n2.forward([self.ct,np.append(x,self.ht)])
        return self.ht

    def backward(self, dJdy,  optimizer = None):
        dJdx_group = self.n2.backward(dJdy, optimizer)
        [[dJdx1,dJdx2],[dJdx3,dJdx4]] = self.n1.backward(dJdx_group[0], optimizer)
        dJdx = dJdx_group[1]+dJdx1+dJdx3+dJdx4
        return dJdx[:self.input_size]

class LSTM(GenericLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        vars = ['xh','c']
        xh = Input(vars,'xh')
        c = Input(vars,'c')
        Wi = MatrixWeight(input_size+output_size,output_size)
        Wf = MatrixWeight(input_size+output_size,output_size)
        Wc = MatrixWeight(input_size+output_size,output_size)
        bf = Weight(output_size)
        bi = Weight(output_size)
        bc = Weight(output_size)
        self.ct_net = ComputationalGraphLayer(
            Sigmoid(Wf*xh+bf)*c+
            Sigmoid(Wi*xh+bi)*Tanh(Wc*xh+bc)
        )
        Wo = MatrixWeight(input_size+output_size,output_size)
        bo = Weight(output_size)
        self.ht_net = ComputationalGraphLayer(
            Tanh(c)*(Wo*xh+bo)
        )

        self.ct = np.zeros(output_size)
        self.ht = np.zeros(output_size)

    def forward(self, x, update = False):
        xh = np.hstack([x,self.ht])
        self.ct = self.ct_net.forward([xh,self.ct])
        self.ht = self.ht_net.forward([xh,self.ct])
        return self.ht

    def backward(self, dJdy,  optimizer = None):
        dJdx_group = self.ht_net.backward(dJdy, optimizer)

        dJdx = self.ct_net.backward(dJdx_group[0], optimizer)
        return dJdx