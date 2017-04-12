import numpy as np
from reinforcement import Ace,Ase,Agent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from layers import GenericLayer
from classicnetwork import Kohonen

time_end = 100000
time_step = 0.01

#States with position of cart and ball
# ball_x = 5
# ball_y = 5
# ball_vel_x = 2
# ball_vel_y = 2
# cart_x = 6
# cart_vel_x = 4
# states_num = ball_x*ball_y*ball_vel_x*ball_vel_y*cart_x*cart_vel_x
#######################################

#States with position of cart and ball
ball_x = 5
ball_y = 5
ball_vx = 5
cart_x = 6
cart_vx = 6
states_num = ball_x*ball_y*ball_vx*cart_x*cart_vx
# states_num = 600
#######################################

#States with position of cart and ball
# ball_states = 5
# cart_statas = 6
# states_num = cart_statas*ball_states
#######################################

#States with differential position only
# states_num = 2#6.0
#######################################


load = 1
interval = 1

if load == 1:
    kon = GenericLayer.load('kon.net')
    agent = GenericLayer.load('agent.net')
    # ace = GenericLayer.load('ace.net')
    # ase = GenericLayer.load('ase.net')
else:
    kon = Kohonen(
        6,
        states_num,
        (30,20,False),
        2
    )
    agent = Agent(states_num, 3, policy='gaussian', learning_rate=0.8, gamma=0.99)
    # ace = Ace(states_num,0.8)
    # ase = Ase(states_num,0.8)

def combine_states(state_list, state_dim):
    val = 0
    power = 1
    for ind,state in enumerate(state_list):
        val = val+state*power
        power = power*state_dim[ind]

    tot_state = np.zeros(np.prod(state_dim))
    tot_state[val] = 1
    return tot_state

class Ball():
    def __init__(self):
        self.a = np.array([0,-9.81])
        self.v = np.array([np.random.rand(1)[0]*4.0-2.0,np.random.rand(1)[0]])
        self.p = np.array([np.random.rand(1)[0]*3+1,4.0])
        self.lose = 0
        self.catch = 0
        self.side = 0

    def step(self, dt, cart):
        self.catch = 0
        self.lose = 0
        self.side = 0
        self.v = self.v + dt*self.a
        self.p = self.p + dt*self.v
        if self.p[0] > cart.p[0]-cart.w/2 and self.p[0] < cart.p[0]+cart.w/2 and self.p[1] < cart.p[1]:
            self.v[1] = -self.v[1]
            self.v[0] += cart.v[0]*0.1
            self.catch = 1
            self.p[1] = cart.p[1]+0.1

        if self.p[0] < 0 or self.p[0] >= 4.9999:
            self.v[0] = -self.v[0]
            self.side = 1
            # self.lose = 1
            if self.p[0] < 0:
                self.p[0] = 0
            if self.p[0] >= 4.9999:
                self.p[0] = 4.9998

        if self.p[1] < 0:
            self.lose = 1

        if  self.p[1] >= 4.9999:
            self.p[1] = 4.9998

class Cart():
    def __init__(self):
        self.m = 0.005
        self.a = np.array([0,0])
        self.v = np.array([0.0,0.0])
        self.p = np.array([int(np.random.rand(1)[0]*5.0),0.5])
        self.w = 1.0

    def step(self, dt, command_x, command_y = 0):
        self.a = np.array([command_x/self.m,command_y/self.m])
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
    for ind,time in enumerate(np.linspace(0,time_end,time_end/time_step)):
        # print time
        ball.step(time_step, cart)

        # state = kon.forward(np.array([ball.p[0],ball.p[1],ball.v[0],ball.v[1],cart.p[0],cart.v[0]]),True)
        stateballx = int(ball.p[0]/5.0*ball_x)
        statebally = int(ball.p[1]/5.0*ball_y)
        stateballvx = int(np.round(ball.v[0])+2)
        if stateballvx <= 0:
            stateballvx = 0
        if stateballvx >= ball_vx-1:
            stateballvx = ball_vx-1

        valcartx = int((cart.p[0]+cart.w/2.0))
        if valcartx <= 0:
            statecartx = 0
        elif valcartx >= 5:
            statecartx = 5
        else:
            statecartx = valcartx

        statecartlvx = int(np.round(cart.v[0])+2)
        if statecartlvx <= 0:
            statecartlvx = 0
        if statecartlvx >= ball_vx-1:
            statecartlvx = ball_vx-1

        # print stateballx,statebally,stateballvx,statediff,statediff_vx
        state = combine_states([stateballx,statebally,stateballvx,statecartx,statecartlvx],[ball_x,ball_y,ball_vx,cart_x,cart_vx])
        # # print state

        # States with position of cart and ball
        # stateball = int(ball.p[0]/5.0*ball_states)
        # valcart = int((cart.p[0]+cart.w/2.0))
        # if valcart <= 0:
        #     statecart = 0
        # elif valcart >= 5:
        #     statecart = 5
        # else:
        #     statecart = valcart
        # # print statecart
        # state = combine_states([stateball,statecart],[ball_states,cart_statas])
        #########################################

        #States with differential position only
        # dist = (ball.p[0]-cart.p[0])
        # state = np.zeros(states_num)
        # if dist < 0:
        #     state[0] = 1 # only 2 state
        #     # if dist <= -cart.w/2:
        #     #     state[0] = 1
        #     # else:
        #     #     state[1] = 1
        #     # elif dist <= -cart.w/2-0.15:
        #     #     state[1] = 1
        #     # elif dist <= -cart.w/4:
        #     #     state[2] = 1
        # else:
        #     state[1] = 1 # only 2 state
        #     # if dist >= cart.w/2:
        #     #     state[2] = 1
        #     # else:
        #     #     state[1] = 1
        #     # elif dist >= cart.w/2+0.15:
        #     #     state[1] = 1
        #     # elif dist >= cart.w/4:
        #     #     state[2] = 1
        ##########################################
        if ball.lose == 0:
            if ball.catch:
                # agent.reinforcement(state,100.0*catches)
                agent.reinforcement(np.argmax(state),100.0*catches)
                catches += 1
                # print 'catch'

                if catches > 10:
                    cart = Cart()
                    ball = Ball()
                    catches = 0

            # if ball.side == 1:
            #     agent.reinforcement(np.argmax(state),-25.0)
            # ind_command = agent.reinforcement(state,0)
            ind_command = agent.reinforcement(np.argmax(state),0)
            command = 0
            if ind_command == 0:
                command = -1
            elif ind_command == 1:
                command = 1
            elif ind_command == 2:
                command = 0

            # agent.reinforcement(np.argmax(state),-np.abs(command)/10.0)
            cart.step(time_step, command)
        else:
            # print 'boing'
            # agent.reinforcement(state,-50.0)
            agent.reinforcement(np.argmax(state),-50.0)
            catches = 0
            cart = Cart()
            ball = Ball()
        if int(ind%1000) == 0:

            #States with position of cart and ball
            # print agent.Q.reshape(ball_states,cart_statas)*np.array([[1,-1,-1,-1,-1,-1],
            #                                                        [1,1,-1,-1,-1,-1],
            #                                                        [1,1,1,-1,-1,-1],
            #                                                        [1,1,1,1,-1,-1],
            #                                                        [1,1,1,1,1,-1],])>0
            # print ace.W.reshape(ball_states,cart_statas)
            ##########################################

            # States with differential position only
            # print (ace.W,ase.W)
            # print agent.Q[agent.Q>0]
            # print agent.Q
            # print agent.Q[agent.Q>1].size/float(states_num*2)*100
            #########################################

            # ase.save('ase.net')
            # ace.save('ace.net')
            kon.save('kon.net')
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
