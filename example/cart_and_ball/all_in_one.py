import numpy as np

##(1)----------------------------------Global variables--------------------------------------
#World init varibles
vertical_desksize = 5.0
horizontal_desksize = 5.0
gravity = -9.81
interval = 1

#Ball init variables
max_start_velocity = 5

#Cart init varibles
cart_mass = 0.01
cart_width = 0.5
viscous_friction = 0.1
cart_height = 0.5
time_end = 100
time_step = 0.01
##------------------------------------------------------------------------------------------

##(2)-----------------------------------Game Dynamic----------------------------------------
class Ball():
    def __init__(self):
        self.a = np.array([0, gravity])
        self.v = np.array([np.random.rand(1)[0]*max_start_velocity*2-max_start_velocity,np.random.rand(1)[0]])
        self.p = np.array([np.random.rand(1)[0]*(horizontal_desksize-2)+1,4.0])
        self.lose = 0
        self.catch = 0
        self.side = 0

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

        if self.p[0] < 0 or self.p[0] >= horizontal_desksize:
            self.v[0] = -self.v[0]

        if self.p[1] < 0:
            self.lose = 1

        if  self.p[1] >= vertical_desksize:
            self.p[1] = vertical_desksize

class Cart():
    def __init__(self):
        self.m = cart_mass
        self.a = np.array([0,0])
        self.v = np.array([0.0,0.0])
        self.p = np.array([int(np.random.rand(1)[0]*horizontal_desksize),cart_height])
        self.w = cart_width

    def step(self, dt, command_x, command_y = 0):
        self.a = np.array([command_x/self.m,command_y/self.m]) - (self.v*viscous_friction)/self.m
        self.v = self.v + dt*self.a
        self.p = self.p + dt*self.v
        if self.p[0] <= 0.0 or self.p[0] >= horizontal_desksize:
            self.v[0] = 0.0
            if self.p[0] <= 0:
                self.p[0] = 0
            if self.p[0] >= horizontal_desksize:
                self.p[0] = horizontal_desksize
##------------------------------------------------------------------------------------------

##(3)-----------------------------------Qlearning Agent-------------------------------------
class Agent():
    def __init__(self, state_size, action_size, learning_rate = 0.1, gamma = 0.95, policy = 'esp-greedy', epsilon = 0.3, sigma = 1):
        self.Q = np.random.randn(action_size, state_size)
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma
        self.y = 0
        self.x = 0
        self.policies = {
            'greedy' : self.greedy,
            'esp-greedy' : self.eps_greedy,
            'gaussian' : self.gaussian
        }
        self.policy = self.policies.get(policy)

    @staticmethod
    def to_one_hot_vect(ind, num_classes):
        on_hot_vect = np.zeros(num_classes)
        on_hot_vect[ind] = 1
        return on_hot_vect

    def greedy(self, x):
        self.y = np.argmax(self.Q[:,x])
        return self.y

    def eps_greedy(self, x):
        if np.random.rand(1,1) < self.epsilon:
            self.y = int(np.random.rand(1,1)*self.Q[:,x].size)
        else:
            self.y = np.argmax(self.Q[:,x])
        return self.y

    def gaussian(self, x):
        self.y = np.argmax(self.Q[:,x]+np.random.normal(0,self.sigma,size=self.Q[:,x].size))
        return self.y

    def forward(self, x, update = False):
        self.x = np.argmax(x)
        return self.to_one_hot_vect(self.policy(self.x),self.action_size)

    def reinforcement(self, x, r):
        self.Q[self.y,self.x] += self.learning_rate*(r+self.gamma*np.max(self.Q[:,np.argmax(x)])-self.Q[self.y,self.x])
        return self.forward(x)

states_num = 2
actions_num = 2
agent = Agent(states_num, actions_num, policy='gaussian', learning_rate=0.2, gamma=0.95)
##------------------------------------------------------------------------------------------

##(4)-----------------------------------Game Logic------------------------------------------
def data_gen():
    cart = Cart()
    ball = Ball()
    catches = 0
    for ind,time in enumerate(np.linspace(0,time_end,time_end/time_step)):
        ball.step(time_step, cart)

        #state selection
        dist = (ball.p[0]-cart.p[0])
        state = np.zeros(states_num)
        if dist < 0:
            state[0] = 1
        else:
            state[1] = 1

        #ball action
        if ball.lose == 0:
            #select action of the cart
            if ball.catch:
                #give a big reward to the cart if it catches the ball
                agent.reinforcement(state,100.0*catches)
                catches += 1

                if catches > 10:
                    cart = Cart()
                    ball = Ball()
                    catches = 0

            #choose the action of the cart
            ind_command = np.argmax(agent.reinforcement(state,0))
            command = -1 if ind_command == 0 else 1

            #move the cart
            cart.step(time_step, command)
        else:
            #give a punishment to the cart if it has lost the ball
            agent.reinforcement(state,-50.0)

            catches = 0
            cart = Cart()
            ball = Ball()
        if int(ind%200) == 0:
            print '%f Have I learned to catch the ball in all the conditions? %s'\
                  %(time,np.reshape(agent.Q*np.array([[1.0,-1.0],[-1.0,1.0]]) > 0,(1,4)))

        yield (cart,ball)
##------------------------------------------------------------------------------------------

##(5)-----------------------------------Graphics--------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(1)
ax = plt.subplot(111)

ax.grid()
ballPoint, = ax.plot(0, 0, color='r', marker='o', linestyle='none')
cartLine, = ax.plot([0,1],[0,0],  color='g', marker='s')
ax.set_xlim(0,horizontal_desksize)
ax.set_ylim(0,vertical_desksize)

def run(data):
    # pass
    ballPoint.set_xdata(data[1].p[0])
    ballPoint.set_ydata(data[1].p[1])
    cartLine.set_xdata([data[0].p[0]-data[0].w/2,data[0].p[0]+data[0].w/2])
    cartLine.set_ydata([data[0].p[1],data[0].p[1]])

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=interval, repeat=False)
plt.show()
##------------------------------------------------------------------------------------------
