import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
from itertools import izip

from layers import LinearLayer, SoftMaxLayer, ReluLayer, SigmoidLayer, UnitStepLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from network import Sequential, Parallel
from trainer import Trainer
from printers import Printer2D

num_classes = 10
name_net = "mnist.net"
p = Printer2D()


def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols).astype(np.float)

    return img, np.array(p.to_one_hot_vect(lbl,num_classes))


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


train_data, train_targets = read(dataset = "training", path = "./mnist")
test_data, test_targets = read(dataset = "testing", path = "./mnist")

if os.path.isfile(name_net):
    print "Load Network"
    f = open(name_net, "r")
    model = pickle.load(f)

    err = 0
    for ind in range(len(train_data)):
        if np.argmax(model.forward(train_data[ind])) != np.argmax(train_targets[ind]):
            #print str(err)+' '+str(np.argmax(model.forward(train_data[ind])))+' '+str(np.argmax(train_targets[ind]))
            err += 1
    print (1.0-err/float(len(train_data)))*100.0


    err = 0
    for ind in range(len(test_data)):
        if np.argmax(model.forward(test_data[ind])) != np.argmax(test_targets[ind]):
            err += 1
    print (1.0-err/float(len(test_data)))*100.0


else:
    print "New Network"
    model = Sequential([
        LinearLayer(784, 10, weights='random'),
        # SigmoidLayer(),
        # LinearLayer(50, 50, weights='random'),
        # SigmoidLayer(),
        # ReluLayer(),
        # LinearLayer(8, 8, weights='random'),
        # SigmoidLayer(),
        # LinearLayer(50, 10, weights='random'),
        # ReluLayer(),
        # SoftMaxLayer()
    ])

trainer = Trainer(show_training = True)

J_list, dJdy_list = trainer.learn_minibatch(
    model = model,
    input_data = train_data,
    target_data = train_targets,
    # loss = NegativeLogLikelihoodLoss(),
    loss = CrossEntropyLoss(),
    # loss = SquaredLoss(),
    # optimizer = StocaticGradientDescent(learning_rate=0.5),
    optimizer = SGDMomentum(learning_rate=0.1, momentum=0.01),
            # optimizer=AdaGrad(learning_rate=0.9),
            # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
    epochs = 20,
    batches_num = 200)

# J_list, dJdy_list = trainer.learn(
#     model = model,
#     input_data = train_data,
#     target_data = train_targets,
#     # loss = NegativeLogLikelihoodLoss(),
#     loss = CrossEntropyLoss(),
#     # loss = SquaredLoss(),
#     optimizer = StocaticGradientDescent(learning_rate=0.2),
#     # optimizer = SGDMomentum(learning_rate=0.2, momentum=0.5),
#             # optimizer=AdaGrad(learning_rate=0.9),
#             # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
#     epochs = 10)

f = open(name_net, "w")
pickle.dump(model,f)

plt.figure(3)
plt.title('Errors History (J)')
plt.plot(xrange(len(J_list)), J_list, color='red')
# plt.ylim([0, 2])
plt.xlabel('Epoch')

#
plt.figure(4)
plt.title('Loss Gradient History (dJ/dy)')
plt.plot(xrange(len(dJdy_list)), dJdy_list, color='orange')
plt.xlabel('Epoch')
#
plt.show()