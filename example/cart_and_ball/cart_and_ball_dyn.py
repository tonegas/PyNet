import numpy as np

#World init varibles
vertical_desksize = 5.0
horizontal_desksize = 5.0
gravity = -9.81

#Ball init variables
max_start_velocity = 2

#Cart init varibles
cart_mass = 0.005
cart_width = 1.0
viscous_friction = 0.1
cart_height = 0.5

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
