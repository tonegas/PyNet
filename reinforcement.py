import numpy as np

from genericlayer import GenericLayer
from layers import define_weights, TanhLayer
from network import Sequential


class Ase(GenericLayer):
    def __init__(self, state_size, delta, weights = 'zeros', learning_rate=0.1, sigma = 1):
        self.input_size = state_size
        self.W = define_weights(weights, state_size, 1)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.delta = delta
        self.e = 0

    def forward(self, x, update = False):
        self.x = x
        # self.y = np.sign(self.W.dot(x)+np.random.normal(0,self.sigma))
        self.y = self.W.dot(x)+np.random.normal(0,self.sigma)
        self.e = self.delta*self.e+(1-self.delta)*self.y*self.x
        return self.y

    # def backward(self, dJdy, optimizer = None):
        # pass

    def reinforcement(self, r): #r>1 successo e r<1 fallimento
        self.W += self.learning_rate * r * self.e


class Ace(GenericLayer):
    def __init__(self, state_size, delta, weights ='random', learning_rate = 0.1, gamma = 0.95,):
        self.input_size = state_size
        self.W = define_weights(weights, state_size, 1)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.delta = delta
        self.p = 0
        self.dp = 0
        self.e = 0

    def forward(self, x, update = False):
        self.x = x
        self.y = self.gamma*self.W.dot(x)-self.p
        self.e = self.delta*self.e+(1-self.delta)*self.x
        self.dp = self.W.dot(x)-self.p
        self.p = self.W.dot(x)
        return self.y

    def reinforcement(self, r):  #r>1 successo e r<1 fallimento
        self.W += self.learning_rate * (r + self.dp) * self.e


import matplotlib.pyplot as plt
import matplotlib.animation as animation

time_end = 1000
time_step = 0.01

# ball_states = 5.0
# cart_statas = 6.0
# states_num = cart_statas*ball_states

#States with differential position only
states_num = 2.0#6.0
#######################################


load = 1
interval = 1

if load == 1:
    # ace = GenericLayer.load('ace.net')
    ase = GenericLayer.load('ase.net')
else:
    # ace = Ace(states_num,0.2)
    ase = Ase(states_num,0.2)

def combine_states(state_list, state_dim):
    val = state_list[0]
    for ind,state in enumerate(state_list[1:]):
        val = state+val*state_dim[ind-1]

    tot_state = np.zeros(np.prod(state_dim))
    tot_state[val] = 1
    return tot_state

class Ball():
    def __init__(self):
        self.a = np.array([0,-9.81])
        self.v = np.array([np.random.rand(1)[0]*4.0-2.0,np.random.rand(1)[0]])
        self.p = np.array([np.random.rand(1)[0]*5,4.0])
        self.lose = 0
        self.catch = 0

    def step(self, dt, cart):
        self.catch = 0
        self.lose = 0
        self.v = self.v + dt*self.a
        self.p = self.p + dt*self.v
        if self.p[0] > cart.p[0]-cart.w/2 and self.p[0] < cart.p[0]+cart.w/2 and self.p[1] < cart.p[1]:
            self.v[1] = -self.v[1]
            self.v[0] += cart.v[0]*0.1
            self.catch = 1
            self.p[1] = cart.p[1]+0.1

        if self.p[0] < 0 or self.p[0] >= 4.9999:
            # self.v[0] = -self.v[0]
            if self.p[0] < 0:
                self.p[0] = 4.9998
            if self.p[0] >= 4.9999:
                self.p[0] = 0

        if self.p[1] < 0:
            self.v[1] = -self.v[1]*1.01
            self.lose = 1

        if  self.p[1] >= 4.9999:
            self.p[1] = 4.9998

class Cart():
    def __init__(self):
        self.m = 0.01
        self.a = np.array([0,0])
        self.v = np.array([0.0,0.0])
        self.p = np.array([int(np.random.rand(1)[0]*5.0),0.5])
        self.w = 0.5

    def step(self, dt, command):
        self.a = np.array([command/self.m,0])
        self.v = self.v*0.9 + dt*self.a
        self.p = self.p + dt*self.v
        # self.p[0] = self.p[0] + command*5.0/cart_statas
        # if self.p[0]-self.w/2 <= 0 or self.p[0]+self.w/2 >= 5:
        #     self.v[0] = 0
        #     if self.p[0]-self.w/2 <= 0:
        #         self.p[0] = self.w/2
        #     if self.p[0]+self.w/2 >= 5:
        #         self.p[0] = 5-self.w/2

def data_gen(t=0):
    cart = Cart()
    ball = Ball()
    catches = 0
    net = TanhLayer()
    for ind,time in enumerate(np.linspace(0,time_end,time_end/time_step)):
        # print time
        ball.step(time_step, cart)
        if ball.lose == 0:
            if ball.catch:
                catches += 1
                # print 'catch'
                # ase.reinforcement(50)

                #States with differential position only
                ase.reinforcement(10)
                ##########################################

                # ace.reinforcement(100)
                if catches > 10:
                    cart = Cart()
                    ball = Ball()
                    catches = 0



            # stateball = int(ball.p[0]/5.0*ball_states)
            # valcart = int(cart.p[0]/5.0*cart_statas)
            # if valcart <= 0:
            #     statecart = 0
            # elif valcart >= 5:
            #     statecart = 5
            # else:
            #     statecart = valcart

            # print stateball,statecart

            # state = combine_states([stateball,statecart],[ball_states,cart_statas])
            # print state

            #States with differential position only
            dist = (ball.p[0]-cart.p[0])
            state = np.zeros(states_num)
            if dist < 0:
                state[0] = 1
                # if dist <= -cart.w/2:
                #     state[0] = 1
                # else:
                #     state[1] = 1
                # elif dist <= -cart.w/2-0.15:
                #     state[1] = 1
                # elif dist <= -cart.w/4:
                #     state[2] = 1
            else:
                state[1] = 1
                # if dist >= cart.w/2:
                #     state[2] = 1
                # else:
                #     state[1] = 1
                # elif dist >= cart.w/2+0.15:
                #     state[1] = 1
                # elif dist >= cart.w/4:
                #     state[2] = 1
            ##########################################

            command = net.forward(ase.forward(state))
            # print ace.forward(state)
            # ase.reinforcement(ace.forward(state))
            cart.step(time_step, command)
        else:
            # print 'boing'
            ase.reinforcement(-50)
            # ace.reinforcement(-5)

            cart = Cart()
            ball = Ball()
        if int(ind%1000) == 0:
            # print ase.W.reshape(ball_states,cart_statas)

            #States with differential position only
            print ase.W
            ##########################################

            ase.save('ase.net')
            # ace.save('ace.net')
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
