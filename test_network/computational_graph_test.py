import numpy as np

from layers import WeightLayer, SumLayer, MulLayer, ConstantLayer, SigmoidLayer, LinearLayer, WeightMatrixLayer
from network import ParallelGroup, Sequential, MapGroup, SumGroup, MulGroup
from genericlayer import GenericLayer, WithElements


class ComputationalGraphLayer(GenericLayer):
    def __init__(self, operation):
        self.net = operation.get()

    def forward(self, x, update = False):
        return self.net.forward(x)

class SelectVariable(GenericLayer):
    def __init__(self, variables, var):
        self.ind = variables[var]

    def forward(self, x_group, update = False):
        if type(x_group) is list:
            return x_group[self.ind]
        else:
            return x_group

# class ListVariable():
#     def __init__(self, variables):
#         self.elements = {}
#         self.dict_variables = {}
#         for ind, var in enumerate(variables):
#             self.dict_variables[var] = ind
#
#         for var in variables:
#             self.elements[var] = SelectVariable(self.dict_variables, var)
#
#     def get(self, label):
#         return self.elements[label]


#(x+y)*(y*y+3)
xx = np.array([1,2,3])
yy = np.array([5,2,-1])
xy = [xx,yy]

# lv = ListVariable(['x','y'])
#
# n = Sequential(
#         ParallelGroup(
#             Sequential(
#                 ParallelGroup(
#                     lv.get('x'),
#                     lv.get('y')
#                 ),SumLayer
#             ),
#             Sequential(
#                 ParallelGroup(
#                     Sequential(
#                         ParallelGroup(
#                             lv.get('y'),
#                             lv.get('y'),
#                         ),MulLayer
#                     ),
#                     ConstantLayer(np.array([3,3,3]))
#                 ),SumLayer
#             )
#         ),MulLayer
#     )
#
# print n.forward(input)


class Operation(object):
    def __init__(self):
        self.net = None

    def __add__(self, other):
        o = Operation()
        o.net = Sequential(
            ParallelGroup(
                self.get(),
                other.get()
            ),SumLayer
        )
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
        o = Operation()
        o.net = Sequential(
            ParallelGroup(
                self.get(),
                other.get()
            ),MulLayer
        )
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
        super(MatrixWeight,self).__init__()
        self.net = WeightMatrixLayer(*args, **kwargs)

    def __mul__(self, other):
        o = Operation()
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
        self.net = SelectVariable(dict_variables, variable)

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

lv = {'x':0,'y':1}
a = Weight(3, weights = np.array([3.0,2.0,1.0]))
b = Weight(3, weights = np.array([2.0,2.0,2.0]))
c = Weight(3, weights = np.array([5.0,6.0,3.0]))
x = Input(lv,'x')
y = Input(lv,'y')

cc = ComputationalGraphLayer(Sigmoid(a*x)**2+b*y+c)

print xy
print cc.forward(xy)

#classic neuron

w1 = MatrixWeight(3,3, weights = np.array([[3.0,2.0,1.0],[3.0,2.0,1.0],[3.0,2.0,1.0]]))
b = Weight(3)

nx = Input({'x':0},'x')
ccc = ComputationalGraphLayer(
    Sigmoid(w1*nx+b)
)

print ccc.forward(np.array([1.0,2.0,3.0]))

# f = Sequential(ccc, LinearLayer(3, 1, weights = np.array([1.0,1.0,3.0,1.0])))
# print f.forward(xx)
#
#

# nn = Sequential(ParallelGroup(GenericLayer,GenericLayer),MulLayer)

# print nn.forward(np.array([1,2,3]))

#
# i = Sequential(ii.get('x'),SumLayer)
# # print i.forward([x,y])
# ii = Sequential(
#         ParallelGroup(
#                 Sequential(MapGroup(GenericLayer,GenericLayer),MulLayer),
#                 ConstantLayer(np.array([3,3,3]))
#         ),SumLayer
#     )
# # print ii.forward([x,y])
#
# Sequential(
#     ,
#     i,ii,
# )
#
# o = Sequential(
#         ParallelGroup(
#             i,ii
#         ),MulLayer
#     )
#
# c = ComputationalGraphLayer(
#     #(a*x**2)+(b*x)
#     x+x+y
# )

# print c.forward(np.array([5]))