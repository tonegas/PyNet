import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, SigmoidLayer, UnitStepLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from network import Sequential
from trainer import Trainer
from printers import Printer2D

classes = ["o","v","x","."]
colors = ['r', 'g', 'b', 'y', 'o']
num_classes = 2
p = Printer2D()
plt.close()

def gen_data():
    n = 100

    # N clusters:
    data, targets = datasets.make_classification(
        n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_classes=num_classes, class_sep=1.0, n_clusters_per_class=1)

    # Circles:
    # data, targets = datasets.make_circles(
    #     n_samples=n, shuffle=True, noise=0.05, random_state=None, factor=0.1)

    # Moons:
    # data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05)

    train_data, test_data = data[:n / 2], data[n / 2:]
    train_targets, test_targets = targets[:n / 2], targets[n / 2:]

    return train_data, train_targets, test_data, test_targets


class Perceptron(unittest.TestCase):
    def test_Perceptron(self):
        train_data, train_targets, test_data, test_targets = gen_data()


        model = Sequential([
            LinearLayer(2, 1, weights='random'),
            SigmoidLayer()
        ])

        y = np.zeros(train_data.size)
        for i, (x,target) in enumerate(izip(train_data, train_targets)):
            y[i] = int(np.round(model.forward(x)))

        p.print_data(1, train_data, p.to_one_hot_vect(train_targets,2), 2, ['gray','gray'], classes)
        p.print_data(1, train_data, p.to_one_hot_vect(y,2), 2, colors, ['x','x'])
        plt.title('Before Training')

        trainer = Trainer()

        J_list, dJdy_list = trainer.learn(
            model = model,
            input_data = train_data,
            target_data = train_targets,
            loss = SquaredLoss(),
            optimizer = StocaticGradientDescent(learning_rate=0.01),
                    # optimizer=MomentumSGD(learning_rate=0.1, momentum=0.9),
                    # optimizer=AdaGrad(learning_rate=0.9),
                    # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
            epochs = 30)

        y1 = np.zeros(train_data.size)
        for i, (x,target) in enumerate(izip(train_data, train_targets)):
            y1[i] = int(np.round(model.forward(x)))

        y2 = np.zeros(test_data.size)
        for i, (x,target) in enumerate(izip(test_data, test_targets)):
            y2[i] = int(np.round(model.forward(x)))

        p.print_data(2, train_data, p.to_one_hot_vect(train_targets,2), 2, ['gray','gray'], classes)
        p.print_data(2, train_data, p.to_one_hot_vect(y1,2), 2, colors, ['x','x'])
        p.print_data(2, test_data, p.to_one_hot_vect(y2,2), 2, colors, ['.','.'])
        plt.title('After Training')

        p.print_model(100, model, test_data)

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