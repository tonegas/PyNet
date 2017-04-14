import numpy as np
from standart_network.vanilla import Vanilla
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

# v = GenericLayer.load_or_create(
#     'vanilla.net',
#     Vanilla(
#         vocab_size,vocab_size,100
#     )
# )

v = Vanilla(
        vocab_size,vocab_size,100
    )

x = to_one_hot_vect(char_to_ix['b'],vocab_size)
# print len(x)
# print v.forward(x)
# print v.backward(x)

epochs = 100

display = ShowTraining(epochs)

trainer = Trainer(show_training = True, show_function=display.show)

train = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[0:-1]]
target = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[1:]]

while True:
    J, dJdy = trainer.learn(
        v,
        zip(train,target),
        NegativeLogLikelihoodLoss(),
        # GradientDescent(learning_rate=0.0001),
        # GradientDescentMomentum(learning_rate=0.0001,momentum=0.001),
        AdaGrad(learning_rate=0.001),
        epochs,
        batch_size = len(train)
        #batch_size = 1
    )

    str = ''
    x = to_one_hot_vect(char_to_ix['c'],vocab_size)
    for i in range(50):
        y = v.forward(x)
        str += ix_to_char[np.random.choice(range(vocab_size), p=y.ravel())]
        x = to_one_hot_vect(np.argmax(y),vocab_size)

    print str
    v.save('vanilla.net')
    # for p in xrange(data_size-seq_length):
    #     train = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[p:p+seq_length]]
    #     target = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[p+1:p+seq_length+1]]
    #
    #     J, dJdy = trainer.learn_minibatch(
    #         v,
    #         zip(train,target),
    #         NegativeLogLikelihoodLoss(),
    #         GradientDescentMomentum(learning_rate=0.3, momentum=0.8)
    #     )




# for t in xrange(len(inputs)):
#     xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
#     xs[t][inputs[t]] = 1