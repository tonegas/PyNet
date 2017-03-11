import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, SigmoidLayer, UnitStepLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from sequential import Sequential

classes = ["o","v","x","."]
colors = ['r', 'g', 'b', 'y', 'o']
num_class = 2
n = 100

def print_compare(figure, data, targets, output):
    x = range(num_class)
    y = range(num_class)
    x_err = range(num_class)
    y_err = range(num_class)

    for type in range(num_class):
        x[type] = []
        y[type] = []
        x_err[type] = []
        y_err[type] = []

    for type in range(num_class):
        for i, point in enumerate(data):
            if np.argmax(targets[i]) == np.argmax(output[i]):
                x[np.argmax(targets[i])].append(point[0])
                y[np.argmax(targets[i])].append(point[1])
            else:
                x_err[np.argmax(output[i])].append(point[0])
                y_err[np.argmax(output[i])].append(point[1])

    plt.figure(figure)
    for type in range(num_class):
        plt.scatter(x[type], y[type], s=100, color='green', marker=classes[type])
        plt.scatter(x_err[type], y_err[type], s=100, color='red', marker=classes[type])

def print_data(figure, train_data, train_targets, colors, classes):
    x = range(num_class)
    y = range(num_class)
    for type in range(num_class):
        x[type] = [point[0] for i, point in enumerate(train_data) if np.argmax(train_targets[i]) == type]
        y[type] = [point[1] for i, point in enumerate(train_data) if np.argmax(train_targets[i]) == type]

    plt.figure(figure)
    for type in range(num_class):
        plt.scatter(x[type], y[type], s=100, color=colors[type], marker=classes[type])

def gen_data():
    # N clusters:
    # data, targets = datasets.make_classification(
    #     n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_classes=num_class, class_sep=1.0, n_clusters_per_class=1)

    # Circles:
    data, targets = datasets.make_circles(
        n_samples=n, shuffle=True, noise=0.1, random_state=None, factor=0.1)

    # Moons:
    # data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05)

    target_vect = []
    for i,target in enumerate(targets):
        target_vect.append(np.zeros(num_class))
        target_vect[i][target] = 1.0

    train_data, test_data = data[:n / 2], data[n / 2:]
    train_targets, test_targets = target_vect[:n / 2], target_vect[n / 2:]

    return train_data, train_targets, test_data, test_targets


class Perceptron(unittest.TestCase):
    def test_Perceptron(self):
        train_data, train_targets, test_data, test_targets = gen_data()

        model = Sequential([
            LinearLayer(2, 3, weights='random'),
            SigmoidLayer(),
            #LinearLayer(3, 3, weights='random'),
            #SigmoidLayer(),
            LinearLayer(3, num_class, weights='random'),
            SoftMaxLayer()
        ])

        #
        y1 = []
        for i, (x,target) in enumerate(izip(train_data, train_targets)):
            y1.append(model.forward(x))
        #
        y2 = []
        for i, (x,target) in enumerate(izip(test_data, test_targets)):
            y2.append(model.forward(x))

        print y1

        print_compare(11, train_data, train_targets, y1)
        print_compare(11, test_data, test_targets, y2)
        plt.title('Before Training')

        # print_data(1, train_data, train_targets, ['gray','gray','gray','gray'], classes)
        # print_data(1, test_data, test_targets, ['gray','gray','gray','gray'], classes)
        # print_data(1, train_data, y1, colors, ['x','x','x','x'])
        # print_data(1, train_data, y2, colors, ['x','x','x','x'])
        #plt.title('Before Training')
        #
        J_list, dJdy_list = model.learn(
            input_data = train_data,
            target_data = train_targets,
            loss = NegativeLogLikelihoodLoss(),
            #loss = CrossEntropyLoss(),
            #loss = SquaredLoss(),
            #optimizer = StocaticGradientDescent(learning_rate=0.1),
            optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9),
                    # optimizer=AdaGrad(learning_rate=0.9),
                    # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
            epochs = 200)
        #
        y1 = []
        for i, (x,target) in enumerate(izip(train_data, train_targets)):
            y1.append(model.forward(x))
        #
        y2 = []
        for i, (x,target) in enumerate(izip(test_data, test_targets)):
            y2.append(model.forward(x))
        #
        print_compare(12, train_data, train_targets, y1)
        print_compare(12, test_data, test_targets, y2)
        plt.title('After Training')

        # print_data(2, train_data, train_targets, ['gray','gray','gray','gray'], classes)
        # print_data(2, test_data, test_targets, ['gray','gray','gray','gray'], classes)
        # print_data(2, train_data, y1, colors, ['x','x','x','x'])
        # print_data(2, test_data, y2, colors, ['.','.','.','.'])
        # plt.title('After Training')
        #
        plt.figure(3)
        plt.title('Errors History (J)')
        plt.plot(xrange(len(J_list)), J_list, color='red')
        # plt.ylim([0, 2])
        plt.xlabel('Epoch')

        #
        plt.figure()
        plt.title('Loss Gradient History (dJ/dy)')
        plt.plot(xrange(len(dJdy_list)), dJdy_list, color='orange')
        plt.xlabel('Epoch')
        #
        plt.show()