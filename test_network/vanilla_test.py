import numpy as np
from standart_network.vanilla import Vanilla
from layers import SoftMaxLayer
from trainer import Trainer
from losses import CrossEntropyLoss, NegativeLogLikelihoodLoss
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
from utils import to_one_hot_vect
from printers import ShowTraining
from genericlayer import GenericLayer

data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 100
window_size = 25

# v = GenericLayer.load_or_create(
#     'vanilla.net',
#     Vanilla(
#         vocab_size,vocab_size,100
#     )
# )

Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias


v = Vanilla(
      vocab_size,vocab_size,hidden_size,window_size,
      Wxh = Wxh.copy(),
      Whh = Whh.copy(),
      Why = Why.copy(),
      bh = bh.copy(),
      by = by.copy()
    )
# sm = SoftMaxLayer()

# x = to_one_hot_vect(char_to_ix['b'],vocab_size)
# print len(x)
# print v.forward(x)
# print v.backward(x)

epochs = 50

display = ShowTraining(epochs)

trainer = Trainer(show_training = True, show_function=display.show)

train = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[0:-1]]
target = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[1:]]


# J, dJdy = trainer.learn_window(
#     v,
#     zip(train[:5],target[:5]),
#     NegativeLogLikelihoodLoss(),
#     #CrossEntropyLoss(),
#     AdaGrad(learning_rate=1e-1),
# )
# print J

# J, dJdy = trainer.learn_window(
#     v,
#     zip(train[:5],target[:5]),
#     NegativeLogLikelihoodLoss(),
#     AdaGrad(learning_rate=0.001),
# )
# print J

while True:
    J, dJdy = trainer.learn_throughtime(
        v,
        zip(train,target),
        # CrossEntropyLoss(),
        NegativeLogLikelihoodLoss(),
        # GradientDescent(learning_rate=0.0001),
        # GradientDescentMomentum(learning_rate=0.0001,momentum=0.001),
        AdaGrad(learning_rate=0.001),
        epochs
    )
    str = ''
    x = to_one_hot_vect(char_to_ix['c'],vocab_size)
    for i in range(50):
        y = v.forward(x)
        str += ix_to_char[np.random.choice(range(vocab_size), p=y.ravel())]
        x = to_one_hot_vect(np.argmax(y),vocab_size)
    print str

# print [ix_to_char[np.argmax(t)] for t in train]
# print [ix_to_char[np.argmax(t)] for t in target]
#
# while True:
#     # J, dJdy = trainer.learn_throghtime(
#     #     v,
#     #     zip(train,target),
#     #     NegativeLogLikelihoodLoss(),
#     #     # GradientDescent(learning_rate=0.0001),
#     #     # GradientDescentMomentum(learning_rate=0.0001,momentum=0.001),
#     #     AdaGrad(learning_rate=0.001),
#     #     epochs
#     # )
#     J, dJdy = trainer.learn_window(
#         v,
#         zip(train[:5],target[:5]),
#         NegativeLogLikelihoodLoss(),
#         AdaGrad(learning_rate=0.001),
#     )
#
#     str = ''
#     x = to_one_hot_vect(char_to_ix['c'],vocab_size)
#     for i in range(50):
#         y = v.forward(x)
#         str += ix_to_char[np.random.choice(range(vocab_size), p=y.ravel())]
#         x = to_one_hot_vect(np.argmax(y),vocab_size)
#
#     print str
#     v.save('vanilla.net')
#     # for p in xrange(data_size-seq_length):
#     #     train = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[p:p+seq_length]]
#     #     target = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[p+1:p+seq_length+1]]
#     #
#     #     J, dJdy = trainer.learn_minibatch(
#     #         v,
#     #         zip(train,target),
#     #         NegativeLogLikelihoodLoss(),
#     #         GradientDescentMomentum(learning_rate=0.3, momentum=0.8)
#     #     )




# for t in xrange(len(inputs)):
#     xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
#     xs[t][inputs[t]] = 1