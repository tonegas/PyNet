import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
from itertools import izip

from layers import LinearLayer, TanhLayer, SoftMaxLayer, ReluLayer, SigmoidLayer, HeavisideLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from network import Sequential, Parallel
from trainer import Trainer
from printers import Printer2D
from classicnetwork import Hopfield

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

    return zip(img, np.array(p.to_one_hot_vect(lbl,num_classes)))


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

def img_compare(image1,image2):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()

    ax = fig.add_subplot(1,2,1)
    imgplot = ax.imshow(image1, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    ax1 = fig.add_subplot(1,2,2)
    imgplot1 = ax1.imshow(image2, cmap=mpl.cm.Greys)
    imgplot1.set_interpolation('nearest')
    ax1.xaxis.set_ticks_position('top')
    ax1.yaxis.set_ticks_position('left')

    pyplot.show()



train = read(dataset = "training", path = "./mnist")
test = read(dataset = "testing", path = "./mnist")

def test_results(model,train,test):
    err = 0
    for i,(img,target) in enumerate(train):
        print i
        if np.argmax(model.forward(img)) != np.argmax(target):
            #print str(err)+' '+str(np.argmax(model.forward(train_data[ind])))+' '+str(np.argmax(train_targets[ind]))
            err += 1
    print (1.0-err/float(len(train)))*100.0

    err = 0
    for (img,target) in test:
        #print str(np.argmax(model.forward(test_data[ind])))+' '+str(np.argmax(test_targets[ind]))
        if np.argmax(model.forward(img)) != np.argmax(target):
            err += 1
    print (1.0-err/float(len(test)))*100.0

if os.path.isfile(name_net):
    print "Load Network"
    f = open(name_net, "r")
    model = pickle.load(f)
    test_results(model,train,test)

else:
    print "New Network"
    model = Sequential([
        LinearLayer(784, 10, weights='random'),
        # SigmoidLayer(),
        # LinearLayer(30, 30, weights='random'),
        # SigmoidLayer(),
        # ReluLayer(),
        # LinearLayer(30, 10, weights='random'),
        # SigmoidLayer(),
        # LinearLayer(50, 10, weights='random'),
        # ReluLayer(),
        # SoftMaxLayer()
    ])

mean_val = [np.zeros(784) for i in range(10)]
tot_val = np.zeros(10)
for x,t in train:
    mean_val[np.argmax(t)] += x
    tot_val[np.argmax(t)] += 1

for i in range(10):
    # mean_val[i] = mean_val[i]/tot_val[i]
    mean_val[i] = np.sign(mean_val[i]-125.0)
    # print mean_val[i]
    # show(mean_val[i].reshape(28,28))

n = Hopfield(784)
for i in [0,1,4]:
    n.save_state(mean_val[i])

# print np.sign(train[0][0]-100.0)

for (x,t) in train:
    y = n.forward(np.sign(x-125.0))
    img_compare(x.reshape(28,28),y.reshape(28,28))

#test_results(n,train,test)

exit()

trainer = Trainer(show_training = True)


J_list, dJdy_list = trainer.learn_minibatch(
    model = model,
    train = train,
    # loss = NegativeLogLikelihoodLoss(),
    loss = CrossEntropyLoss(),
    # loss = SquaredLoss(),
    # optimizer = StocaticGradientDescent(learning_rate=0.3),
    optimizer = SGDMomentum(learning_rate=0.95, momentum=0.95),
            # optimizer=AdaGrad(learning_rate=0.9),
            # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
    epochs = 5,
    batch_size = 50)


# J_list, dJdy_list = trainer.learn(
#     model = model,
#     input_data = (train_data-np.mean(train_data))/np.exp(np.max(train_data)),
#     target_data = train_targets,
#     # loss = NegativeLogLikelihoodLoss(),
#     loss = CrossEntropyLoss(),
#     # loss = SquaredLoss(),
#     # optimizer = StocaticGradientDescent(learning_rate=0.5),
#     optimizer = SGDMomentum(learning_rate=0.5, momentum=0.8),
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