import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, SigmoidLayer, UnitStepLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from sequential import Sequential
from trainer import Trainer
from printers import Printer2D

classes = ["o","v","x","."]
colors = ['r', 'g', 'b', 'y', 'o']
num_classes = 4
n = 100
p = Printer2D()
plt.close()

def gen_data(n):
    # N clusters:
    data, targets = datasets.make_classification(
        n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_classes=num_classes, class_sep=3.0, n_clusters_per_class=1)

    # Circles:
    # data, targets = datasets.make_circles(
    #     n_samples=n, shuffle=True, noise=0.05, random_state=None, factor=0.1)

    # Moons:
    # data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05)

    target_vect = p.to_one_hot_vect(targets,num_classes)

    train_data, test_data = data[:n / 2], data[n / 2:]
    train_targets, test_targets = target_vect[:n / 2], target_vect[n / 2:]

    return train_data, train_targets, test_data, test_targets


class Perceptron(unittest.TestCase):
    def test_Perceptron(self):
        max_data = 1000
        all_range = range(10,max_data,400)
        all_J_list = np.zeros(len(all_range))
        train_data_all, train_targets_all, test_data_all, test_targets_all = gen_data(max_data)

        p.print_data(2,train_data_all,train_targets_all,num_classes,colors,classes)

        for ind,n in enumerate(all_range):
            train_data = train_data_all[:n]
            train_targets = train_targets_all[:n]
            test_data = test_data_all[:n]
            test_targets = test_targets_all[:n]


            model = Sequential([
                LinearLayer(2, 5, weights='random'),
                SigmoidLayer(),
                #LinearLayer(3, 3, weights='random'),
                #SigmoidLayer(),
                LinearLayer(5, num_classes, weights='random'),
                SoftMaxLayer()
            ])

            trainer = Trainer()

            #
            J_list, dJdy_list = trainer.learn(
                model = model,
                input_data = train_data,
                target_data = train_targets,
                loss = NegativeLogLikelihoodLoss(),
                #loss = CrossEntropyLoss(),
                #loss = SquaredLoss(),
                #optimizer = StocaticGradientDescent(learning_rate=0.1),
                optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9),
                        # optimizer=AdaGrad(learning_rate=0.9),
                        # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
                epochs = 50)
            #
            y1 = []
            for i, (x,target) in enumerate(izip(train_data, train_targets)):
                y1.append(model.forward(x))
            #
            y2 = []
            for i, (x,target) in enumerate(izip(test_data, test_targets)):
                y2.append(model.forward(x))

        #

            p.compare_data(ind+5, train_data, train_targets, y1, num_classes, classes)
            p.compare_data(ind+5, test_data, test_targets, y2, num_classes, classes)
            plt.title('After Training sample='+str(n))

            # plt.figure(ind+5)
            # plt.title('Errors History (J) sample='+str(n))
            # plt.plot(xrange(len(J_list)), J_list, color='red')
            # # plt.ylim([0, 2])
            # plt.xlabel('Epoch')
            #p.print_model(100, model, train_data)

            all_J_list[ind] = np.min(J_list)
        #
        plt.figure(1)
        plt.title('Errors History (J)')
        plt.plot(all_range, all_J_list, color='blue')
        # plt.ylim([0, 2])
        plt.xlabel('Number of sample')

        #
        plt.show()