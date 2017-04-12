import numpy as np

# def to_one_hot_vect(vect, num_classes):
#     on_hot_vect = []
#     for i,target in enumerate(vect):
#         on_hot_vect.append(np.zeros(num_classes))
#         on_hot_vect[i][target] = 1
#     return on_hot_vect

def to_one_hot_vect(ind, num_classes):
    on_hot_vect = np.zeros(num_classes)
    on_hot_vect[ind] = 1
    return on_hot_vect

def define_weights(weights, input_size, output_size):
    if type(weights) == str:
        if weights == 'random':
            weights_val = np.random.rand(output_size, input_size)
        elif weights == 'norm_random':
            weights_val = (np.random.rand(output_size, input_size)-0.5)/input_size
        elif weights == 'ones':
            weights_val = np.ones([output_size, input_size])
        elif weights == 'zeros':
            weights_val = np.zeros([output_size, input_size])
        else:
            raise Exception('Type not correct!')
    elif type(weights) == np.ndarray or type(weights) == np.matrixlib.defmatrix.matrix:
        weights_val = weights.reshape(output_size, input_size)
    else:
        raise Exception('Type not correct!')
    if input_size == 1:
        return weights_val.reshape(output_size)
    elif output_size == 1:
        return weights_val.reshape(input_size)
    else:
        return weights_val