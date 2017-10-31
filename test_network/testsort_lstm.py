

import numpy as np
from standart_network.lstm import LSTMNet
from layers import SoftMaxLayer, LinearLayer
from trainer import Trainer
from losses import CrossEntropyLoss, NegativeLogLikelihoodLoss
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
from utils import to_one_hot_vect, SharedWeights
from printers import ShowTraining
from genericlayer import GenericLayer
from network import Sequential

data = open('CV.2016.tex', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

train = []
target = []
for ind in range(5000):
    o = map(int,np.random.rand(10)*100+1)
    train.append(np.hstack([o,np.zeros(9)]))
    target.append(np.hstack([np.zeros(9),np.sort(o)]))

vocab_size = 100
hidden_size = 50
window_size = 19

Wi = SharedWeights('gaussian', vocab_size+hidden_size, hidden_size, L2=0.0001)
Wf = SharedWeights('gaussian', vocab_size+hidden_size, hidden_size, L2=0.0001)
Wc = SharedWeights('gaussian', vocab_size+hidden_size, hidden_size, L2=0.0001)
Wo = SharedWeights('gaussian', vocab_size+hidden_size, hidden_size, L2=0.0001)
bi = SharedWeights('zeros', 1, hidden_size)
bf = SharedWeights('zeros', 1, hidden_size)
bc = SharedWeights('zeros', 1, hidden_size)
bo = SharedWeights('zeros', 1, hidden_size)

load = 0
if load:
    lstm = GenericLayer.load('lstm.net')
else:
    l = LSTMNet(vocab_size, hidden_size, Wi=Wi, Wf=Wf, Wc=Wc, Wo=Wo, bi=bi, bf=bf, bc=bc, bo=bo)
    lstm = Sequential(
        l,
        LinearLayer(hidden_size,vocab_size),
    )

sm = SoftMaxLayer()

# lstm.on_message('init_nodes',20)
#
# x = to_one_hot_vect(char_to_ix['b'],vocab_size)
# print len(x)
# for i in range(20):
#     print lstm.forward(x,update = True)
#
# print lstm.backward(x)

epochs = 100

opt = AdaGrad(learning_rate=0.15, clip=5)

display = ShowTraining(epochs_num = epochs, weights_list = {'Wi':l.Wi, 'Wf':l.Wf, 'Wc':l.Wc, 'Wo':l.Wo, 'bf':l.bf, 'bi':l.bi, 'bc':l.bc, 'bo':l.bo})

def functionPlot(*args):
    lstm.save('lstm.net')
    str = ''
    x = to_one_hot_vect(char_to_ix['c'],vocab_size)
    for i in range(200):
        y = sm.forward(lstm.forward(x))
        str += ix_to_char[np.random.choice(range(vocab_size), p=y.ravel())]
        x = to_one_hot_vect(np.argmax(y),vocab_size)
    print str
    display.show(*args)

trainer = Trainer(show_training = True, show_function = functionPlot)

train = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[0:-1]]
target = [to_one_hot_vect(char_to_ix[ch],vocab_size) for ch in data[1:]]

J, dJdy = trainer.learn_throughtime(
    lstm,
    zip(train,target),
    CrossEntropyLoss(),
    opt,
    1,
    window_size
)

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
        lstm,
        zip(train,target),
        CrossEntropyLoss(),
        # NegativeLogLikelihoodLoss(),
        # GradientDescent(learning_rate=0.01),
        # GradientDescentMomentum(learning_rate=0.01,momentum=0.5),
        opt,#,clip=100.0),
        epochs,
        window_size
    )

