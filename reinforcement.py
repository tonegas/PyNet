import numpy as np

from genericlayer import GenericLayer
from layers import define_weights


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
        self.y = np.sign(self.W.dot(x)+np.random.normal(0,self.sigma))
        self.e = self.delta*self.e+(1-self.delta)*self.y*self.x
        return self.y

    def reinforcement(self, r): #r>1 success e r<1 fail
        self.W += self.learning_rate * r * self.e


class Ace(GenericLayer):
    def __init__(self, state_size, delta, weights ='random', learning_rate = 0.1, gamma = 0.95):
        self.input_size = state_size
        self.W = define_weights(weights, state_size, 1)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.delta = delta
        self.p = 0
        self.e = 0

    def forward(self, x, update = False):
        self.x = x
        self.y = self.gamma*self.W.dot(x)-self.p
        self.e = self.delta*self.e+(1-self.delta)*self.x # always positive
        self.p = self.W.dot(x)
        self.reinforcement(0)
        return self.y

    def reinforcement(self, r):  #r>1 success e r<1 fail
        self.W += self.learning_rate * (r + self.y) * self.e

class AseAce(GenericLayer):
    def __init__(self, state_size, delta, weights = 'zeros', learning_rate=0.1, sigma = 1, gamma = 0.95):
        self.ase = Ase(state_size, delta, weights, learning_rate, sigma)
        self.ace = Ace(state_size, delta, weights, learning_rate, gamma)

    def forward(self, x, update = False):
        self.x = x
        self.ase.reinforcement(self.ace.forward(x))
        return self.ase.forward(x)

    def reinforcement(self, r):
        self.ace.reinforcement(r)
        self.ase.reinforcement(r+self.ace.forward(self.x))


class Agent(GenericLayer):
    def __init__(self, state_size, action_size, learning_rate = 0.1, gamma = 0.95, policy = 'esp-greedy', epsilon = 0.3, sigma = 1):
        self.Q = define_weights('zeros', state_size, action_size)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma
        self.y = 0
        self.x = 0
        self.policies = {
            'greedy' : self.greedy,
            'esp-greedy' : self.eps_greedy,
            'gaussian' : self.gaussian,
            'softmax' : self.softmax
        }
        self.policy = self.policies.get(policy)

    def greedy(self, x):
        self.x = x
        self.y = np.argmax(self.Q[:,x])
        return self.y

    def eps_greedy(self, x):
        self.x = x
        if np.random.rand(1,1) < self.epsilon:
            self.y = int(np.random.rand(1,1)*self.Q[:,x].size)
        else:
            self.y = np.argmax(self.Q[:,x])
        return self.y

    def gaussian(self, x):
        self.x = x
        self.y = np.argmax(self.Q[:,x]+np.random.normal(0,self.sigma,size=self.Q[:,x].size))
        return self.y

    def softmax(self, x):
        raise Exception('Not Implemented!')

    def forward(self, x, update = False):
        return self.policy(x)

    def reinforcement(self, x, r):
        self.Q[self.y,self.x] += self.learning_rate*(r+self.gamma*np.max(self.Q[:,x])-self.Q[self.y,self.x])
        return self.policy(x)

