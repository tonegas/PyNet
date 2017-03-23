import numpy as np
from itertools import izip

class Trainer():
    def __init__(self, depth = 1, show_training = False):
        self.depth = depth
        self.show_training = show_training

    def learn_one(self, model, x, t, loss, optimizer):
        y = model.forward(x, True)
        J = loss.loss(y,t)
        dJdy = loss.dJdy_gradient(y,t)
        model.backward(dJdy, optimizer)
        return J, dJdy

    def learn(self, model, train, loss, optimizer, epochs, batch_size = 1, validation = None):
        J_train_list = np.zeros(epochs)
        J_validation_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        num_sample = len(train)
        batches_num = num_sample/batch_size
        # print batches_num
        for epoch in range(epochs):
            np.random.shuffle(train)
            train_vect = np.array_split(train, batches_num)
            # print train_vect
            for batch in train_vect:
                this_batch_size = len(batch)
                # print this_batch_size
                optimizer.store = True
                for i,(x,t) in enumerate(batch):
                    # print 'x'+str(x)
                    y = model.forward(x, True)
                    # print 'y'+str(y)
                    J = loss.loss(y,t)

                    dJdy = loss.dJdy_gradient(y,t)

                    if this_batch_size == i+1:
                        optimizer.store = False

                    model.backward(dJdy, optimizer)

                    J_train_list[epoch] += np.linalg.norm(J)/num_sample
                    dJdy_list[epoch] += np.linalg.norm(dJdy)/num_sample


            if validation:
                for x,t in validation:
                    J_validation_list[epoch] += np.linalg.norm(loss.loss(model.forward(x),t))/num_sample

                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_validation_list[epoch])

            else:
                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        if validation:
            return J_train_list, dJdy_list, J_validation_list
        return J_train_list, dJdy_list

    def on_line_learn(self, model, train, loss, optimizer, epochs, validation = None):
        J_train_list = np.zeros(epochs)
        J_validation_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        for epoch in range(epochs):
            np.random.shuffle(train)
            for (x,t) in train:
                J, dJdy = self.learn_one(model, x, t, loss, optimizer)
                J_train_list[epoch] += np.sum(np.power(J,2))
                dJdy_list[epoch] += np.linalg.norm(dJdy)

            if validation:
                for x,t in validation:
                    J_validation_list[epoch] += np.sum(np.power(loss.loss(model.forward(x),t),2))

                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_validation_list[epoch])

            else:
                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        if validation:
            return J_train_list, dJdy_list, J_validation_list
        return J_train_list, dJdy_list

    # def learn(self, model, train, loss, optimizer, epochs, batch_size, test = None):
    #     J_train_list = np.zeros(epochs)
    #     J_test_list = np.zeros(epochs)
    #     dJdy_list = np.zeros(epochs)
    #     num_sample = len(train)
    #     batches_num = num_sample/batch_size
    #     for epoch in range(epochs):
    #         np.random.shuffle(train)
    #         train_vect = np.array_split(train, batches_num)
    #         for batch in train_vect:
    #             mean_x = np.zeros(batch[0][0].shape)
    #             mean_dJdy = np.zeros(batch[0][1].shape)
    #             train_batch_size = len(batch)
    #             # print train_batch_size
    #             for x,t in batch:
    #                 # print 'x'+str(x)
    #                 y = model.forward(x)
    #                 # print 'y'+str(y)
    #                 J = loss.loss(y,t)
    #
    #                 dJdy = loss.dJdy_gradient(y,t)
    #                 # print 'dJdy'+str(mean_dJdy)
    #
    #                 mean_x += x/float(train_batch_size)
    #                 mean_dJdy += dJdy/float(train_batch_size)
    #                 # print dJdy
    #                 J_train_list[epoch] += np.sqrt(np.sum(np.power(J,2)))/float(num_sample)
    #                 dJdy_list[epoch] += np.mean(dJdy)
    #
    #             model.forward_and_update(mean_x)
    #             model.backwark_and_update(mean_dJdy, optimizer, self.depth)
    #
    #         if test:
    #             tests_size = len(test)
    #             for x,t in test:
    #                 J_test_list[epoch] += np.sqrt(np.sum(np.power(loss.loss(model.forward(x),t),2)))/tests_size
    #
    #             if self.show_training:
    #                 print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_test_list[epoch])
    #
    #         else:
    #             if self.show_training:
    #                 print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])
    #
    #     if test:
    #         return J_train_list, dJdy_list, J_test_list
    #     return J_train_list, dJdy_list
