import numpy as np

from layers import WeightLayer, SumLayer, MulLayer, ConstantLayer, SigmoidLayer, \
    WeightMatrixLayer, NegativeLayer, SelectVariableLayer
from network import ParallelGroup, Sequential, SumGroup
from genericlayer import GenericLayer, WithElements

class Operation(object):
    def __init__(self):
        self.net = None

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        o = Operation()
        o.net = Sequential(
            self.get(),
            NegativeLayer()
        )
        return o

    def __add__(self, other):
        if isinstance(other, Input) or isinstance(other, Operation):
            o = Operation()
            o.net = Sequential(
                ParallelGroup(
                    self.get(),
                    other.get()
                ),SumLayer
            )
        elif isinstance(other, int) or isinstance(other, float):
            o = Operation()
            o.net = Sequential(
                ParallelGroup(
                    self.get(),
                    ConstantLayer(np.array([other]).astype(float))
                ),SumLayer
            )
        else:
            raise Exception('Type is not supported!')
        return o

    def __pow__(self, other):
        o = Operation()
        o.net = Sequential(
            ParallelGroup(
                [self.get() for i in range(other)]
            ),MulLayer
        )
        return o

    def __mul__(self, other):
        if isinstance(other, Input) or isinstance(other, Operation):
            o = Operation()
            o.net = Sequential(
                ParallelGroup(
                    self.get(),
                    other.get()
                ),MulLayer
            )
        elif isinstance(other, int) or isinstance(other, float):
            o = Operation()
            o.net = Sequential(
                ParallelGroup(
                    self.get(),
                    ConstantLayer(np.array([other]).astype(float))
                ),MulLayer
            )
        else:
            raise Exception('Type is not supported!')
        return o

    def get(self):
        return self.net


class Weight(Operation):
    def __init__(self, *args, **kwargs):
        super(Weight,self).__init__()
        self.net = WeightLayer(*args, **kwargs)

    def get(self):
        return self.net

class MatrixWeight(Operation):
    def __init__(self, *args, **kwargs):
        self.a = args
        self.b = kwargs
        super(MatrixWeight,self).__init__()
        self.net = WeightMatrixLayer(*args, **kwargs)

    def __mul__(self, other):
        o = MatrixWeight(*self.a, **self.b)
        o.net = Sequential(
            other.get(),
            self.get()
        )
        return o

    def get(self):
        return self.net

class Input(Operation):
    def __init__(self, dict_variables, variable):
        super(Input,self).__init__()
        self.net = SelectVariableLayer(dict_variables, variable)

    def get(self):
        return self.net

class Sigmoid(Operation):
    def __init__(self, operation):
        super(Sigmoid,self).__init__()
        self.net = Sequential(
            operation.get(),
            SigmoidLayer
        )

    def get(self):
        return self.net