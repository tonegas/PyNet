import numpy as np
import collections
import random

import layers
import utils



class Ase(layers.GenericLayer):
    def __init__(self, state_size, delta, weights = 'zeros', learning_rate=0.1, sigma = 1):
        self.input_size = state_size
        self.W = utils.define_weights(weights, state_size, 1)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.delta = delta
        self.e = 0

    def forward(self, x, update = False):
        self.x = x
        self.y = np.sign(self.W.dot(x)+np.random.normal(0,self.sigma))
        self.e = self.delta*self.e+(1-self.delta)*self.y*self.x
        return utils.to_one_hot_vect(self.y/2.0+1.0,2)

    def reinforcement(self, x, r): #r>1 success e r<1 fail
        self.W += self.learning_rate * r * self.e
        return self.forward(x)


class Ace(layers.GenericLayer):
    def __init__(self, state_size, delta, weights ='zeros', learning_rate = 0.1, gamma = 0.95):
        self.input_size = state_size
        self.W = utils.define_weights(weights, state_size, 1)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.delta = delta
        self.p = 0
        self.e = 0
        self.y = 0

    def forward(self, x, update = False):
        return self.step(x,0)

    def reinforcement(self, x, r):  #r>1 success e r<1 fail
        return self.step(x,r)

    def step(self, x, r):
        self.x = x
        self.y = self.gamma*self.W.dot(x)-self.p
        self.e = self.delta*self.e+(1-self.delta)*self.x # always positive
        self.p = self.W.dot(x)
        self.W += self.learning_rate * (r + self.y) * self.e
        return self.y

class AseAce(layers.GenericLayer):
    def __init__(self, state_size, delta, weights = 'zeros', learning_rate=0.1, sigma = 1, gamma = 0.95):
        self.ase = Ase(state_size, delta, weights, learning_rate, sigma)
        self.ace = Ace(state_size, delta, weights, learning_rate, gamma)

    def forward(self, x, update = False):
        self.x = x
        return self.ase.reinforcement(x, self.ace.reinforcement(x, 0))

    def reinforcement(self, x, r):
        self.x = x
        return self.ase.reinforcement(x, r+self.ace.reinforcement(x, r))


class Agent(layers.GenericLayer):
    def __init__(self, state_size, action_size, learning_rate = 0.1, gamma = 0.95, policy = 'esp-greedy', epsilon = 0.3, sigma = 1):
        self.Q = utils.define_weights('zeros', state_size, action_size)
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
            'gaussian' : self.gaussian,
            'softmax' : self.softmax
        }
        self.policy = self.policies.get(policy)

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

    def softmax(self, x):
        raise Exception('Not Implemented!')

    def forward(self, x, update = False):
        self.x = np.argmax(x)
        return utils.to_one_hot_vect(self.policy(self.x),self.action_size)

    def reinforcement(self, x, r):
        self.Q[self.y,self.x] += self.learning_rate*(r+self.gamma*np.max(self.Q[:,np.argmax(x)])-self.Q[self.y,self.x])
        return self.forward(x)

#Generic Agent
#When the agent receive a reward, it performs a training.
class GenericAgent(layers.GenericLayer):
    def __init__(self, model, action_size, memory_size, pole):
        self.model = model
        self.action_size = action_size
        self.memory_size = memory_size
        self.states_history = collections.deque(maxlen = memory_size)
        self.command_history = collections.deque(maxlen = memory_size)
        self.e = np.tile(np.exp(-pole*np.linspace(0,1,memory_size)),(action_size,1)).T
        self.command = np.zeros(action_size)

        # self.target_history = np.zeros_like(self.e)

    def set_training_options(self, trainer, loss, optimizer):
        self.trainer = trainer
        self.loss = loss
        self.optimiser = optimizer

       # def reinforcement(self, state, reinforcement):
    #     self.states_history.append(state)
    #     self.command_history.append(self.command)
    #     self.target_history = np.roll(self.target_history,1,axis=0)
    #     self.target_history[0,:] = np.zeros([1,self.output_size])
    #     if reinforcement != 0:
    #         self.target_history += reinforcement*np.multiply(self.e,np.array(self.command_history))
    #         # print zip(np.argmax(self.command_history,axis=1),self.target_history)
    #
    #         if len(self.command_history) >= self.memory_size:
    #             self.trainer.learn_minibatch(
    #                 self.net,
    #                 zip(self.states_history,self.target_history),
    #                 self.loss,
    #                 self.optimiser,
    #             )
    #         #self.command_history.clear()
    #
    #     self.command = to_one_hot_vect(np.argmax(self.net.forward(state)),self.output_size)
    #     return self.command

    def forward(self, x, update = False):
        self.states_history.append(x)
        self.command_history.append(self.command)
        self.command = utils.to_one_hot_vect(np.argmax(self.model.forward(x)),self.action_size)
        return self.command

    def reinforcement(self, x, r):
        self.states_history.append(x)
        self.command_history.append(self.command)
        if len(self.command_history) >= self.memory_size:
            # print np.argmax(self.command_history,axis=1)
            if r != 0:
                self.target = r*np.multiply(self.e,np.array(self.command_history))
                # self.target = r*np.array(self.command_history)
                self.trainer.learn_minibatch(
                    self.model,
                    zip(self.states_history,self.target),
                    self.loss,
                    self.optimiser,
                )

        self.command = utils.to_one_hot_vect(np.argmax(self.model.forward(x,True)),self.action_size)
        return self.command

    def clear(self):
        self.command_history.clear()


#Appunti
#Q function e' la rete quindi io mi posso salvare lo stato che e' un input
#e poi l'uscita mi da l'azione migliore trovata fino a quel momento
class DeepAgent(layers.GenericLayer):
    def __init__(self, Q, Q_hat, replay_memory_size, minibatch_size = 100, learning_rate = 0.1, gamma = 1, policy = 'eps-greedy', epsilon = 0.3, sigma = 0.5):
        self.Q = Q
        self.Q_hat = Q_hat
        self.D_size = replay_memory_size
        self.D = collections.deque(maxlen = replay_memory_size)
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma
        self.x = 0
        self.Q_out = 0
        self.action = 0
        self.policies = {
            'greedy' : self.greedy,
            'eps-greedy' : self.eps_greedy,
            'gaussian' : self.gaussian,
            'softmax' : self.softmax
        }
        self.policy = self.policies.get(policy)

    def greedy(self, x):
        self.Q_out =  self.Q.forward(x)
        self.action = np.argmax(self.Q_out)
        return self.action

    def eps_greedy(self, x):
        self.Q_out = self.Q.forward(x)
        if np.random.rand(1,1) < self.epsilon:
            self.action = int(np.random.rand(1,1)*self.Q_out.size)
        else:
            self.action = np.argmax(self.Q_out)
        return self.action

    def gaussian(self, x):
        self.Q_out = self.Q.forward(x)
        self.action = np.argmax(self.Q_out+np.random.normal(0,self.sigma,size=self.Q_out.size))
        return self.action

    def softmax(self, x):
        raise Exception('Not Implemented!')

    def set_training_options(self, trainer, loss, optimiser):
        self.trainer = trainer
        self.loss = loss
        self.optimiser = optimiser

    #sei in uno stato, valuti tutte le possibili mosse che possono essere fatte
    #per ogni mossa chiami la rete e ottieni la probabilita di vittoria
    #te la salvi in un vettore azioni
    def forward(self, x, update = False):
        self.x = x
        self.action = self.policy(self.x)
        return self.action

    def reinforcement(self, x, r, done):
        self.D.append((self.x, self.action, r, x, done))
        J_train_list = 0
        dJdy_list = 0
        if len(self.D) < self.minibatch_size:
            return J_train_list, dJdy_list

        minibatch = random.sample(self.D, self.minibatch_size)
        #y is the target of the loss function
        y = []
        Q_out = []
        states = []
        for state, action, reward, next_state, done, in minibatch:
            states.append(state)
            #y_val = self.Q_hat.forward(state)
            Q_out_val = self.Q.forward(state)#*utils.to_one_hot_vect(action,self.Q_out.size)
            Q_out.append(Q_out_val)
            if done == False:
                # yj = r(t) + gamma * max_action(Q_hat(x(t+1),action))
                yj = reward + self.gamma * np.max(self.Q.forward(next_state))
            else:
                # yj = r(t)
                yj = reward
            # yj = self.Q_hat.forward(next_state)
            y_val = Q_out_val.copy()
            y_val[action] = yj
            # print (Q_out_val,y_val,action,yj)
            # y_val = yj*utils.to_one_hot_vect(action,self.Q_out.size)
            y.append(y_val)

        # J_train_list, dJdy_list = self.trainer.learn_minibatch(
        #     self.Q,
        #     zip(states,y),
        #     self.loss,
        #     self.optimiser,
        # )

            J = self.loss.loss(Q_out_val,y_val)/self.minibatch_size
            dJdy = self.loss.dJdy_gradient(Q_out_val,y_val)/self.minibatch_size
            self.Q.backward(dJdy, self.optimiser)

            J_train_list += np.linalg.norm(J)
            dJdy_list += np.linalg.norm(dJdy)

        self.optimiser.update_model()


        #print len(states)
        # print (np.array(y),np.array(Q_out))
        # print np.max(np.array(y)-np.array(Q_out))
        #y = [minibatch[2][ind] + self.gamma * np.max(self.Q_hat.forward(minibatch[3][ind])) if done == 0 else minibatch[2][ind] for ind,done in enumerate(minibatch[4])]

        # states = np.random.rand(100,2)
        # y = []
        # for x in states:
        #     y.append(self.Q_hat.forward(x))




        return J_train_list, dJdy_list
