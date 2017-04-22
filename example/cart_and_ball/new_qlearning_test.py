import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cart_and_ball_dyn import Cart, Ball
from layers import GenericLayer
from utils import to_one_hot_vect

time_end = 100000
time_step = 0.01

from trainer import Trainer
from losses import NegativeLogLikelihoodLoss, SquaredLoss
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
import collections

class AddGaussian(GenericLayer):
    def __init__(self, sigma = 1):
        self.sigma = sigma

    def forward(self, x, update = False):
        self.y = x + np.random.normal(0,self.sigma,size=x.size)
        return self.y

    def backward(self, dJdy, optimizer = None):
        return dJdy

class Agent(GenericLayer):
    def __init__(self, net, output_size, memory_size, pole):
        self.net = net
        self.output_size = output_size
        self.memory_size = memory_size
        self.states_history = collections.deque(maxlen = memory_size)
        #self.rewards_history = collections.deque(maxlen = memory_size)
        self.command_history = collections.deque(maxlen = memory_size)
        self.e = np.tile(np.exp(-pole*np.linspace(0,1,memory_size)),(output_size,1)).T
        self.trainer = Trainer(show_training = True)
        self.loss = NegativeLogLikelihoodLoss()
        self.optimiser = GradientDescentMomentum(learning_rate = 0.1, momentum = 0.7) #GradientDescent(learning_rate=0.2) #
        self.command = np.zeros(output_size)

    # def reinforcement(self, state, reinforcement):
    #     self.states_history.append(state)
    #     #self.rewards_history.append(reinforcement)
    #     self.command_history.append(self.command)
    #     if len(self.command_history) >= self.memory_size:
    #         # print self.e
    #         # print np.array(self.command_history)
    #         # print reinforcement*np.multiply(self.e,np.array(self.command_history))
    #
    #         if reinforcement != 0:
    #             self.target = reinforcement*np.multiply(self.e,np.array(self.command_history))
    #             # print self.target
    #             # print zip(self.states_history,self.target)
    #             # print np.argmax(self.command_history,axis=1)
    #             # self.trainer.learn(
    #             #     self.net,
    #             #     zip(self.states_history,self.target),
    #             #     self.loss,
    #             #     self.optimiser,
    #             #     1,
    #             #     batch_size = self.memory_size
    #             # )
    #             self.trainer.learn_minibatch(
    #                 self.net,
    #                 zip(self.states_history,self.target),
    #                 self.loss,
    #                 self.optimiser,
    #             )
    #             self.command_history.clear()
    #
    #     self.command = to_one_hot_vect(np.argmax(self.net.forward(state)),self.output_size)
    #     # print (state,self.command)
    #     return self.command

        self.target = np.zeros_like(self.e)
        self.reset = 1

    def reinforcement(self, state, reinforcement):
        self.states_history.append(state)
        self.command_history.append(self.command)
        if reinforcement != 0:
            self.target += reinforcement*np.multiply(self.e,np.array(self.command_history))
            self.reset = 0

        if self.reset == 0 and len(self.command_history) >= self.memory_size:
            self.trainer.learn_minibatch(
                self.net,
                zip(self.states_history,self.target),
                self.loss,
                self.optimiser,
            )
            self.target = np.zeros_like(self.e)
            self.reset = 1
            # self.command_history.clear()

        self.command = to_one_hot_vect(np.argmax(self.net.forward(state)),self.output_size)
        return self.command

#
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
from layers import SoftMaxLayer, LinearLayer, TanhLayer, NormalizationLayer

interval = 1
load = 1

if load == 1:
    agent = GenericLayer.load('agent.net')
    # from printers import Printer2D
    # p = Printer2D()
    # p.print_model(2,agent.net,np.array([[0,0],[5.0,5.0]]))
    # plt.show()
else:
    # norm = NormalizationLayer(
    #     np.array([0.0,0.0,0.0,-3.0,-3.0]),
    #     np.array([5.0,5.0,5.0,3.0,3.0]),
    #     np.array([-1.0,-1.0,-1.0,-1.0,-1.0]),
    #     np.array([1.0,1.0,1.0,1.0,1.0])
    # )
    norm = NormalizationLayer(
        np.array([0.0,0.0,0.0,-3.0]),
        np.array([5.0,5.0,5.0,3.0]),
        np.array([-1.0,-1.0,-1.0,-1.0]),
        np.array([1.0,1.0,1.0,1.0])
    )

    n = Sequential(
        norm,
        LinearLayer(4,3,weights='gaussian'),
        TanhLayer,
        #AddGaussian(1),
        LinearLayer(3,3,weights='gaussian'),
        AddGaussian(1),
        SoftMaxLayer
    )
    agent = Agent(n,3,20,5.0)

def data_gen(t=0):
    cart = Cart()
    ball = Ball()
    catches = 0
    for ind,time in enumerate(np.linspace(0,time_end,time_end/time_step)):
        # print time
        ball.step(time_step, cart)

        state = np.array([ball.p[0],cart.p[0],ball.p[1],ball.v[0]])
        #print state
        if ball.lose == 0:
            if ball.catch:
                agent.reinforcement(state,0.005)
                catches += 1
                # print 'catch'

                if catches > 10:
                    cart = Cart()
                    ball = Ball()
                    catches = 0

            # if ball.side == 1:
            #     agent.reinforcement(np.argmax(state),-0.)
            ind_command = np.argmax(agent.reinforcement(state,0))
            #print ind_command
            command = 0
            if ind_command == 0:
                command = 1
            elif ind_command == 1:
                command = -1
            elif ind_command == 2:
                command = 0

            # if cart.lose == 1:
            #     cart.lose = 0
            #     agent.reinforcement(state,-0.2)

            cart.step(time_step, command)
        else:
            # print 'boing'
            agent.reinforcement(state,-0.3)
            catches = 0
            cart = Cart()
            ball = Ball()
        if int(ind%1000) == 0:
            # print (agent.net.elements[0].W[0][3],agent.net.elements[0].W[1][3],agent.net.elements[0].W[2][3])
            #print agent.net.elements[1].W
            print agent.net.elements[3].W
            agent.save('agent.net')
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
