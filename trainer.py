import numpy as np
from itertools import izip

class Trainer():
    def __init__(self, depth = 1, show_training = False):
        self.depth = depth
        self.show_training = show_training

    def learn_one(self, model, x, t, loss, optimizer):
        y = model.forward_and_update(x,0)
        J = loss.loss(y,t)
        dJdy = loss.dJdy_gradient(y,t)
        model.backward_and_update(dJdy, optimizer)
        return J, dJdy

    def learn(self, model, train, loss, optimizer, epochs, test = None):
        J_train_list = np.zeros(epochs)
        J_test_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        train_size = len(train)
        for epoch in range(epochs):
            # np.random.shuffle(train)
            for (x,t) in train:
                J, dJdy = self.learn_one(model, x, t, loss, optimizer)
                J_train_list[epoch] += np.sqrt(np.sum(np.power(J,2)))/train_size
                dJdy_list[epoch] += np.mean(dJdy)

            if test:
                tests_size = len(test)
                for x,t in test:
                    J_test_list[epoch] += np.sqrt(np.sum(np.power(loss.loss(model.forward(x),t),2)))/tests_size

                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_test_list[epoch])

            else:
                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        if test:
            return J_train_list, dJdy_list, J_test_list
        return J_train_list, dJdy_list

    def learn_minibatch(self, model, train, loss, optimizer, epochs, batch_size, test = None):
        J_train_list = np.zeros(epochs)
        J_test_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        num_sample = len(train)
        batches_num = num_sample/batch_size
        for epoch in range(epochs):
            np.random.shuffle(train)
            train_vect = np.array_split(train, batches_num)
            for batch in train_vect:
                mean_x = np.zeros(batch[0][0].shape)
                mean_dJdy = np.zeros(batch[0][1].shape)
                train_batch_size = len(batch)
                # print train_batch_size
                for x,t in batch:
                    # print 'x'+str(x)
                    y = model.forward(x)
                    # print 'y'+str(y)
                    J = loss.loss(y,t)

                    dJdy = loss.dJdy_gradient(y,t)
                    # print 'dJdy'+str(mean_dJdy)

                    mean_x += x/float(train_batch_size)
                    mean_dJdy += dJdy/float(train_batch_size)
                    # print dJdy
                    J_train_list[epoch] += np.sqrt(np.sum(np.power(J,2)))/float(num_sample)
                    dJdy_list[epoch] += np.mean(dJdy)

                model.forward_and_update(mean_x)
                model.backwark_and_update(mean_dJdy, optimizer, self.depth)

            if test:
                tests_size = len(test)
                for x,t in test:
                    J_test_list[epoch] += np.sqrt(np.sum(np.power(loss.loss(model.forward(x),t),2)))/tests_size

                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_test_list[epoch])

            else:
                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        if test:
            return J_train_list, dJdy_list, J_test_list
        return J_train_list, dJdy_list
