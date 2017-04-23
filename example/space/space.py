import numpy as np

class Ball():
    def __init__(self, start):
        self.a = np.array([0,0])
        self.v = np.array([0,0])
        self.p = start
        self.lose = 0
        self.win = 0

    def step(self, dt, command, obstacles, win):
        self.lose = 0
        self.win = 0
        self.a = command
        self.v = self.v + dt*self.a
        self.p = self.p + dt*self.v
        for ob in obstacles:
            if np.sqrt((self.p[0]-ob[0])**2+(self.p[1]-ob[1])**2) < ob[2]:
                self.lose = 1

        if self.p[0] < 0 or self.p[0] > 5 or self.p[1] > 5 or self.p[1] < 0:
            self.lose = 1

        if np.sqrt((self.p[0]-win[0])**2+(self.p[1]-win[1])**2) < win[2]:
            self.win = 1