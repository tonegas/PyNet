import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import utils
from cart_and_ball_dyn import Cart, Ball
from layers import GenericLayer
from utils import to_one_hot_vect

time_end = 100000
time_step = 0.01

from trainer import Trainer
from losses import HuberLoss, SquaredLoss
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
from qlearning import DeepAgent

# l = LinearLayer(2,5)
# s = SoftMaxLayer()
# n = Sequential(l,s)
#
# x = np.array([1,2])
# t = np.array([0,-0.5,0,0,0])
#
# y = n.forward(x)
# print y
#
# loss = NegativeLogLikelihoodLoss()
#
# J = loss.loss(y,t)
# dJdy = loss.dJdy_gradient(y,t)
# dJdx_softmax = s.backward(dJdy)
#
# print dJdx_softmax
# dJdx = l.backward(dJdx_softmax)
# print dJdx
# l.W -= l.dJdW_gradient(dJdx_softmax)
# print n.forward(np.array([1,2]))

from network import Sequential
from layers import  TanhLayer, LinearLayer, ReluLayer, NormalizationLayer

interval = 0
load = 1

if load == 1:
    agent = GenericLayer.load('deepagent.net')
    W1 = agent.Q.elements[1].W
    # W2 = agent.Q.elements[3].W
    W3 = agent.Q_hat.elements[1].W
    # W4 = agent.Q_hat.elements[3].W
    from printers import Printer2D
    p = Printer2D()
    #p.print_model(2,agent.Q,np.array([[0,0],[5.0,5.0]]))
    p.draw_decision_surface(10,agent.Q,np.array([[0,0],[5.0,5.0]]))
    plt.show()
else:
    # norm = NormalizationLayer(
    #     np.array([0.0,0.0,0.0,-3.0,-3.0]),
    #     np.array([5.0,5.0,5.0,3.0,3.0]),
    #     np.array([-1.0,-1.0,-1.0,-1.0,-1.0]),
    #     np.array([1.0,1.0,1.0,1.0,1.0])
    # )
    # norm = NormalizationLayer(
    #     np.array([0.0,0.0,0.0,-3.0]),
    #     np.array([5.0,5.0,5.0,3.0]),
    #     np.array([-1.0,-1.0,-1.0,-1.0]),
    #     np.array([1.0,1.0,1.0,1.0])
    # )
    norm = NormalizationLayer(
        np.array([0.0,0.0]),
        np.array([5.0,5.0]),
        np.array([0.0,0.0]),
        np.array([1.0,1.0])
    )
    W1 = utils.SharedWeights('gaussian',2+1,2)
    W2 = utils.SharedWeights('gaussian',2+1,3)
    Q = Sequential(
        norm,
        LinearLayer(2,2,weights=W1),
        TanhLayer,
        LinearLayer(2,3,weights=W2),
        # TanhLayer
    )
    W3 = utils.SharedWeights('gaussian',2+1,2)
    W4 = utils.SharedWeights('gaussian',2+1,3)
    # W3 = utils.SharedWeights(np.array([[10.0,-10.0,0.0],[-10.0,10.0,0.0]]),2+1,2)
    #W2 = utils.SharedWeights('gaussian',2+1,2)
    Q_hat = Sequential(
        norm,
        LinearLayer(2,2,weights=W3),
        ReluLayer,
        LinearLayer(2,3,weights=W4),
        # TanhLayer
    )
    #Q, Q_hat, replay_memory_size, minibatch_size = 100, learning_rate = 0.1, gamma = 0.95, policy = 'esp-greedy', epsilon = 0.3
    agent = DeepAgent(Q,Q_hat, 1000, minibatch_size = 100, policy = 'eps-greedy')
    agent.set_training_options(
        Trainer(show_training=True),
        SquaredLoss(),
        # GradientDescent(learning_rate=0.001)
        GradientDescentMomentum(learning_rate=0.1, momentum=0.2, clip=1) #
    )

J_train_list = []
dJdy_list = []
def data_gen(t=0):
    cart = Cart()
    ball = Ball()
    catches = 0
    for ind,time in enumerate(np.linspace(0,time_end,time_end/time_step)):
        # print time

        state = np.array([ball.p[0],cart.p[0]]) #,ball.p[1],ball.v[0]])
        ind_command = agent.forward(state)
        if ind_command == 0:
            command = 1
        elif ind_command == 1:
            command = -1
        elif ind_command == 2:
            command = 0
        # elif ind_command == 3:
        #     command = 0
        cart.step(time_step, command)
        ball.step(time_step, cart)
        state = np.array([ball.p[0],cart.p[0]])

        #print state
        if ball.lose == 0:
            if ball.catch:
                catches += 1
                # print 'catch'

                if catches > 10:
                    cart = Cart()
                    ball = Ball()
                    catches = 0

                J, dJdy = agent.reinforcement(state,1,catches==0)
            else:
                J, dJdy = agent.reinforcement(state,0,False)
            # if ball.side == 1:
            #     agent.reinforcement(np.argmax(state),-0.)



            # if cart.lose == 1:
            #     cart.lose = 0
            #     agent.reinforcement(state,-0.2)
        else:
            J, dJdy = agent.reinforcement(state,-1,True)
            catches = 0
            cart = Cart()
            ball = Ball()

        J_train_list.append(J)
        dJdy_list.append(dJdy)


        # print W1.get()
        if int(ind%1000) == 0:
            print 't:'+str(time)+' J_train:'+str(J_train_list[ind])+' dJdy_list:'+str(dJdy_list[ind])
            #W3.W = W1.W.copy()
            print W1.W
            # W4.W = W2.W.copy()
            #print (np.max(W1.get()),np.max(W2.get()))
            # print agent.net.elements[1].W
            # print np.max(agent.model.elements[1].W)
            agent.save('deepagent.net')
        yield (cart,ball)

fig = plt.figure(1)
ax = plt.subplot(111)

ax.grid()
ballPoint, = ax.plot(0, 0, color='r', marker='o', linestyle='none')
cartLine, = ax.plot([0,1],[0,0],  color='g', marker='s')
ax.set_xlim(0,5)
ax.set_ylim(0,5)

def run(data):
    # pass
    ballPoint.set_xdata(data[1].p[0])
    ballPoint.set_ydata(data[1].p[1])
    cartLine.set_xdata([data[0].p[0]-data[0].w/2,data[0].p[0]+data[0].w/2])
    cartLine.set_ydata([data[0].p[1],data[0].p[1]])

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=interval, repeat=False)
plt.show()

