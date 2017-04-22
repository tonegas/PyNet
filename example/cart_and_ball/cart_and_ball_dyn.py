import numpy as np

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
                # self.p[0] = 4.9998
            if self.p[0] >= 4.9999:
                # self.p[0] = 0
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
        self.lose = 0

    def step(self, dt, command_x, command_y = 0):
        self.a = np.array([command_x/self.m,command_y/self.m])
        self.v = self.v*0.9 + dt*self.a
        self.p = self.p + dt*self.v
        # self.p[0] = self.p[0] + command*5.0/cart_statas
        if self.p[0]<= 0 or self.p[0] >= 5:
            self.lose = 1
            self.v[0] = 0
            if self.p[0] <= 0:
                self.p[0] = 0
            if self.p[0]+self.w/2 >= 5:
                self.p[0] = 5


