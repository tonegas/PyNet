import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets

from layers import LinearLayer, TanhLayer, SoftMaxLayer, ReluLayer, SigmoidLayer, HeavisideLayer, NormalizationLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import GradientDescent, GradientDescentMomentum
from network import Sequential, Parallel
from trainer import Trainer
from printers import Printer2D, ShowTraining
from utils import to_one_hot_vect

classes = ["o","v","x",".","s"]
colors = ['r', 'g', 'b', 'y', 'o']
num_classes = 2
n = 200
epochs = 200
p = Printer2D()


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

    targets = [to_one_hot_vect(target,num_classes) for target in targets]

    train = zip(np.array(data[:n *9/10]).astype(np.float), np.array(targets[:n *9/10]).astype(np.float))
    test = zip(np.array(data[n /10:]).astype(np.float), np.array(targets[n /10:]).astype(np.float))


    return train, test


train, test = gen_data()

validation = train[:n/10]
train = train[n/10+1:]


model = Sequential([
    LinearLayer(2, 20, weights='random'),
    TanhLayer(),
    #SigmoidLayer(),
    # HeavisideLayer(),
    # LinearLayer(10, 20, weights='random'),
    # SigmoidLayer(),
    LinearLayer(20, num_classes, weights='random',L1=0.001),
    # ReluLayer(),
    # SigmoidLayer()
    SoftMaxLayer()
])

# model = Sequential([
#     LinearLayer(2, 5, weights='random'),
#     SigmoidLayer(),
#     #LinearLayer(3, 3, weights='random'),
#     #SigmoidLayer(),
#     LinearLayer(5, 4, weights='random'),
#     # Parallel([
#     #     LinearLayer(5, 1, weights='random'),
#     #     LinearLayer(5, 1, weights='random'),
#     #     LinearLayer(5, 1, weights='random'),
#     #     LinearLayer(5, 1, weights='random'),
#     # ]),
#     # SigmoidLayer(),
#     SoftMaxLayer()
# ])

#
y1 = []
for i, (x,target) in enumerate(train):
    y1.append(model.forward(x))
#
y2 = []
for i, (x,target) in enumerate(test):
    y2.append(model.forward(x))

# p.compare_data(1, train_data, train_targets, y1, num_classes, classes)
# p.compare_data(1, test_data, test_targets, y2, num_classes, classes)
# plt.title('Before Training')

# print_data(1, train_data, train_targets, ['gray','gray','gray','gray'], classes)
# print_data(1, test_data, test_targets, ['gray','gray','gray','gray'], classes)
# print_data(1, train_data, y1, colors, ['x','x','x','x'])
# print_data(1, train_data, y2, colors, ['x','x','x','x'])
#plt.title('Before Training')

display = ShowTraining(epochs_num = epochs)

trainer = Trainer(show_training = True, show_function = display.show)

t = time.time()

J_train_list, dJdy_list, J_validation_list = trainer.learn(
    model = model,
    train = train,
    validation = validation,
    loss = NegativeLogLikelihoodLoss(),
    # loss = CrossEntropyLoss(),
    # loss = SquaredLoss(),
    # optimizer = GradientDescent(learning_rate = 0.15/110),
    optimizer = GradientDescentMomentum(learning_rate=0.005, momentum=0.8),
    batch_size = 100,
    epochs = epochs
)


elapsed = time.time() - t
print 'Training time: '+str(elapsed)

y1 = []
for i, (x,target) in enumerate(train):
    y1.append(model.forward(x))
#
y2 = []
for i, (x,target) in enumerate(test):
    y2.append(model.forward(x))
#

# p.print_model(100, model, [x for (x,t) in train])

p.draw_decision_surface(10, model, test)
p.compare_data(10, train, y1, num_classes, colors, classes)
p.compare_data(10, test, y2, num_classes, colors, classes)
plt.title('After Training')
plt.figure(11)

# print_data(2, train_data, train_targets, ['gray','gray','gray','gray'], classes)
# print_data(2, test_data, test_targets, ['gray','gray','gray','gray'], classes)
# print_data(2, train_data, y1, colors, ['x','x','x','x'])
# print_data(2, test_data, y2, colors, ['.','.','.','.'])
# plt.title('After Training')

raw_input('Press ENTER to exit')
