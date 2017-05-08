import numpy as np

def to_hot_vect(vect, num_classes):
    on_hot_vect = []
    for i,target in enumerate(vect):
        on_hot_vect.append(np.zeros(num_classes))
        on_hot_vect[i][target] = 1
    return on_hot_vect

def to_one_hot_vect(ind, num_classes):
    on_hot_vect = np.zeros(num_classes)
    on_hot_vect[ind] = 1
    return on_hot_vect

def define_weights(weights, input_size = None, output_size = None):
    if type(weights) == str and type(input_size) == int and type(output_size) == int:
        if weights == 'random':
            weights_val = np.random.rand(output_size, input_size)
        elif weights == 'norm-random':
            weights_val = (np.random.rand(output_size, input_size)-0.5)/input_size
        elif weights == 'gaussian':
            weights_val = np.random.randn(output_size, input_size)/input_size
        elif weights == 'ones':
            weights_val = np.ones([output_size, input_size])
        elif weights == 'zeros':
            weights_val = np.zeros([output_size, input_size])
        else:
            raise Exception('Type not correct!')
    elif type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
        weights_val = weights
    else:
        raise Exception('Type not correct!')
    if len(weights_val.shape) == 1:
        return weights_val.reshape(weights_val.shape[0]).copy()
    elif len(weights_val.shape) == 2:
        if weights_val.shape[0] == 1:
            # print weights_val.reshape(weights_val.shape[1])
            return weights_val.reshape(weights_val.shape[1]).copy()
        elif weights_val.shape[1] == 1:
            return weights_val.reshape(weights_val.shape[0]).copy()
        else:
            return weights_val.copy()

class SharedWeights():
    def __init__(self, weights = 'gaussian', input_size = None, output_size = None, L1 = 0.0, L2 = 0.0):
        self.L1 = L1
        self.L2 = L2
        if type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
            self.W = define_weights(weights)
            self.dW = np.zeros_like(self.W)
        elif input_size is not None and output_size is not None:
            self.W = define_weights(weights, input_size, output_size)
            self.dW = np.zeros_like(self.W)
        else:
            raise Exception('Type not correct!')

    @staticmethod
    def get_or_create(weights, input_size = None, output_size = None, L1 = 0.0, L2 = 0.0):
        if isinstance(weights, SharedWeights):
            return weights
        else:
            return SharedWeights(weights, input_size, output_size, L1 = 0.0, L2 = 0.0)

    def T(self):
        out = SharedWeights(self)
        out.W = self.W.T
        out.dW = self.dW.T
        return out

    def get(self):
        return self.W

    def get_dW(self):
        return self.dW
