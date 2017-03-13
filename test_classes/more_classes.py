import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, ReluLayer, SigmoidLayer, UnitStepLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from network import Sequential, Parallel
from trainer import Trainer
from printers import Printer2D

classes = ["o","v","x",".","s"]
colors = ['r', 'g', 'b', 'y', 'o']
num_classes = 2
n = 1000
p = Printer2D()
plt.close()

def gen_data():
    # N clusters:
    # data, targets = datasets.make_classification(
    #     n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_classes=num_classes, class_sep=3.0, n_clusters_per_class=1)

    data, targets = datasets.make_gaussian_quantiles(
        mean=(0,0), cov=1, n_samples=n, n_classes=num_classes)

    # Circles:
    # data, targets = datasets.make_circles(
    #     n_samples=n, shuffle=True, noise=0.1, random_state=None, factor=0.1)

    # Moons:
    # data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05)

    # print data
    # print targets

    target_vect = p.to_one_hot_vect(targets,num_classes)

    train_data, test_data = np.array(data[:n / 2]).astype(np.float), np.array(data[n / 2:]).astype(np.float)
    train_targets, test_targets = np.array(target_vect[:n / 2]).astype(np.float), np.array(target_vect[n / 2:]).astype(np.float)

    return train_data, train_targets, test_data, test_targets


class Perceptron(unittest.TestCase):
    def test_Perceptron(self):
        train_data, train_targets, test_data, test_targets = gen_data()

        model = Sequential([
            LinearLayer(2, 8, weights='random'),
            SigmoidLayer(),
            # LinearLayer(8, 8, weights='random'),
            # SigmoidLayer(),
            LinearLayer(8, num_classes, weights='random'),
            SoftMaxLayer()
        ])

        # model = Sequential([
        #     LinearLayer(2, 5, weights='random'),
        #     SigmoidLayer(),
        #     #LinearLayer(3, 3, weights='random'),
        #     #SigmoidLayer(),
        #     Parallel([
        #         LinearLayer(5, 1, weights='random'),
        #         LinearLayer(5, 1, weights='random'),
        #         LinearLayer(5, 1, weights='random'),
        #         LinearLayer(5, 1, weights='random'),
        #     ]),
        #     # SigmoidLayer(),
        #     SoftMaxLayer()
        # ])

        #
        y1 = []
        for i, (x,target) in enumerate(izip(train_data, train_targets)):
            y1.append(model.forward(x))
        #
        y2 = []
        for i, (x,target) in enumerate(izip(test_data, test_targets)):
            y2.append(model.forward(x))

        p.compare_data(1, train_data, train_targets, y1, num_classes, classes)
        p.compare_data(1, test_data, test_targets, y2, num_classes, classes)
        plt.title('Before Training')

        # print_data(1, train_data, train_targets, ['gray','gray','gray','gray'], classes)
        # print_data(1, test_data, test_targets, ['gray','gray','gray','gray'], classes)
        # print_data(1, train_data, y1, colors, ['x','x','x','x'])
        # print_data(1, train_data, y2, colors, ['x','x','x','x'])
        #plt.title('Before Training')
        #
        trainer = Trainer(depth = 2, show_training = True)

        J_list, dJdy_list = trainer.learn(
            model = model,
            input_data = train_data,
            target_data = train_targets,
            loss = NegativeLogLikelihoodLoss(),
            # loss = CrossEntropyLoss(),
            # loss = SquaredLoss(),
            #optimizer = StocaticGradientDescent(learning_rate=0.1),
            optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9),
                    # optimizer=AdaGrad(learning_rate=0.9),
                    # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
            epochs = 200)

        # J_list, dJdy_list = trainer.learn_minibatch(
        #     model = model,
        #     input_data = train_data,
        #     target_data = train_targets,
        #     loss = NegativeLogLikelihoodLoss(),
        #     # loss = CrossEntropyLoss(),
        #     # loss = SquaredLoss(),
        #     # optimizer = StocaticGradientDescent(learning_rate=0.1),
        #     optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9),
        #             # optimizer=AdaGrad(learning_rate=0.9),
        #             # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
        #     epochs = 200,
        #     batches_num = 200)
        #
        y1 = []
        for i, (x,target) in enumerate(izip(train_data, train_targets)):
            y1.append(model.forward(x))
        #
        y2 = []
        for i, (x,target) in enumerate(izip(test_data, test_targets)):
            y2.append(model.forward(x))
        #
        plt.figure(2)
        p.compare_data(2, train_data, train_targets, y1, num_classes, classes)
        p.compare_data(2, test_data, test_targets, y2, num_classes, classes)
        plt.title('After Training')

        # print_data(2, train_data, train_targets, ['gray','gray','gray','gray'], classes)
        # print_data(2, test_data, test_targets, ['gray','gray','gray','gray'], classes)
        # print_data(2, train_data, y1, colors, ['x','x','x','x'])
        # print_data(2, test_data, y2, colors, ['.','.','.','.'])
        # plt.title('After Training')

        p.print_model(100, model, train_data)
        #
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