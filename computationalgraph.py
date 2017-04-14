import numpy as np

import layers
from network import  Sequential, SequentialMul, SequentialSum, SequentialNegative
from genericlayer import GenericLayer, WithElements

class Op(object):
    def __init__(self,net):
        self.net = net

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return Op(SequentialNegative(self.get()))

    def __add__(self, other):
        if isinstance(other, Input) or isinstance(other, Op):
            if isinstance(self.net, SequentialSum):
                return Op(self.net.add(other.get()))
            return Op(SequentialSum(self.get(),other.get()))

        elif isinstance(other, int) or isinstance(other, float):
            if isinstance(self.net, SequentialSum):
                return Op(self.net.add(layers.ConstantLayer(np.array([other]).astype(float))))
            return Op(SequentialSum(self.get(),layers.ConstantLayer(np.array([other]).astype(float))))

        else:
            raise Exception('Type is not supported!')

    def __pow__(self, other):
        return Op(SequentialMul(*[self.get() for i in range(other)]))

    def __mul__(self, other):
        if isinstance(other, Input) or isinstance(other, Op):
            if isinstance(self.net, SequentialMul):
                return Op(self.net.add(other.get()))
            return Op(SequentialMul(self.get(),other.get()))

        elif isinstance(other, int) or isinstance(other, float):
            if isinstance(self.net, SequentialMul):
                return Op(self.net.add(layers.ConstantLayer(np.array([other]).astype(float))))
            return Op(SequentialMul(self.get(),layers.ConstantLayer(np.array([other]).astype(float))))

        else:
            raise Exception('Type is not supported!')

    def get(self):
        return self.net


class VWeight(Op):
    def __init__(self, *args, **kwargs):
        super(VWeight,self).__init__(layers.VWeightLayer(*args, **kwargs))

class MWeight(Op):
    def __init__(self, *args, **kwargs):
        self.a = args
        self.b = kwargs
        super(MWeight,self).__init__(layers.MWeightLayer(*args, **kwargs))

    def __mul__(self, other):
        o = MWeight(*self.a, **self.b)
        o.net = Sequential(
            other.get(),
            self.get()
        )
        return o

class MWeightBias(Op):
    def __init__(self, *args, **kwargs):
        self.a = args
        self.b = kwargs
        super(MWeightBias,self).__init__(layers.LinearLayer(*args, **kwargs))

    def __mul__(self, other):
        o = MWeight(*self.a, **self.b)
        o.net = Sequential(
            other.get(),
            self.get()
        )
        return o

class Input(Op):
    def __init__(self, dict_variables, variable):
        super(Input,self).__init__(layers.SelectVariableLayer(dict_variables, variable))

class Sigmoid(Op):
    def __init__(self, operation):
        super(Sigmoid,self).__init__(
            Sequential(
                operation.get(),
                layers.SigmoidLayer
            )
        )

class Tanh(Op):
    def __init__(self, operation):
        super(Tanh,self).__init__(
            Sequential(
                operation.get(),
                layers.TanhLayer
            )
        )

class Softmax(Op):
    def __init__(self, operation):
        super(Softmax,self).__init__(
            Sequential(
                operation.get(),
                layers.SoftMaxLayer
            )
        )
