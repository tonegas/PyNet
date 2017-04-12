import numpy as np
from standart_network.vanilla import Vanilla
from trainer import Trainer
from losses import CrossEntropyLoss
from optimizers import GradientDescentMomentum

data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def to_one_hot_vect(vect, num_classes):
    on_hot_vect = []
    for i,target in enumerate(vect):
        on_hot_vect.append(np.zeros(num_classes))
        on_hot_vect[i][target] = 1
    return on_hot_vect

v = Vanilla(
    vocab_size,vocab_size,20
)

seq_length = 10

trainer = Trainer(
    show_training = True
)

counter=0
while True:
    for p in xrange(data_size-seq_length):
        train = to_one_hot_vect([char_to_ix[ch] for ch in data[p:p+seq_length]],vocab_size)
        target = to_one_hot_vect([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]],vocab_size)

        J, dJdy = trainer.learn_minibatch(
            v,
            zip(train,target),
            CrossEntropyLoss(),
            GradientDescentMomentum(learning_rate=0.3, momentum=0.8)
        )

    counter += 1
    if counter%100 == 0:
        str = ''
        x = train[1]
        for i in range(100):
            y = v.forward(x)
            str += ix_to_char[np.argmax(y)]
            x = to_one_hot_vect([np.argmax(y)],vocab_size)[0]

        print str

# for t in xrange(len(inputs)):
#     xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
#     xs[t][inputs[t]] = 1