import numpy as np

class Trainer():
    def __init__(self, show_training = False, show_function = None):
        self.show_training = show_training
        self.show_function = show_function

    def learn_one(self, model, x, t, loss, optimizer):
        y = model.forward(x, True)
        J = loss.loss(y,t)
        dJdy = loss.dJdy_gradient(y,t)
        model.backward(dJdy, optimizer)
        optimizer.update_model()
        return J, dJdy

    def learn_window(self, model, batch, loss, optimizer):
        this_batch_size = len(batch)
        # print this_batch_size
        J_train_list = 0
        dJdy_list = 0
        y_list = []
        for i,(x,t) in enumerate(batch):
            # print model.forward(x, True)
            y = model.forward(x, True)
            y_list.append(y)
            # print 'x'+str(x)
            # print 'y'+str(y)
            # print 't'+str(t)
            J = loss.loss(y,t)
            J_train_list += np.linalg.norm(J)/this_batch_size

        # print '------------------------------------------'
        for i,(x,t) in enumerate(reversed(batch)):
            # print 'x'+str(x)
            # print 'y'+str(y_list[this_batch_size-1-i])
            # print 't'+str(t)
            dJdy = loss.dJdy_gradient(y_list[this_batch_size-1-i],t)

            model.backward(dJdy, optimizer)
            dJdy_list += np.linalg.norm(dJdy)/this_batch_size

        optimizer.update_model()

        return J_train_list, dJdy_list

    def learn_throughtime(self, model, train, loss, optimizer, epochs, test = None):
        J_train_list = np.zeros(epochs)
        J_test_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        train_num = len(train)
        window_size = model.window_size
        # if test is not None:
        #     test_num = len(test)
        batches_num = train_num/window_size
        # print batches_num
        for epoch in range(epochs):
            train_vect = np.array_split(train, batches_num)
            # print train_vect
            for batch in train_vect[:]:
                J, dJdy = self.learn_window(model, batch, loss, optimizer)
                J_train_list[epoch] += J/batches_num
                dJdy_list[epoch] += dJdy/batches_num

            # if test:
            #     for x,t in test:
            #         J_test_list[epoch] += np.linalg.norm(loss.loss(model.forward(x),t))/test_num
            #
            #     if self.show_training:
            #         if self.show_function is not None:
            #             self.show_function(epoch, J_train_list, dJdy_list, J_test_list)
            #         else:
            #             print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_test_list[epoch])
            #
            # else:
            if self.show_training:
                if self.show_function is not None:
                    self.show_function(epoch, J_train_list, dJdy_list)
                else:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        # if test:
        #     return J_train_list, dJdy_list, J_test_list
        return J_train_list, dJdy_list

    def learn_minibatch(self, model, batch, loss, optimizer):
        this_batch_size = len(batch)
        # print this_batch_size
        J_train_list = 0
        dJdy_list = 0
        for i,(x,t) in enumerate(batch):
            # print 'x'+str(x)
            y = model.forward(x, True)
            # print 'y'+str(y)
            J = loss.loss(y,t)

            # print y,t
            dJdy = loss.dJdy_gradient(y,t)

            model.backward(dJdy, optimizer)

            J_train_list += np.linalg.norm(J)/this_batch_size
            dJdy_list += np.linalg.norm(dJdy)/this_batch_size

        optimizer.update_model()

        return J_train_list, dJdy_list

    def learn(self, model, train, loss, optimizer, epochs, batch_size = 1, test = None):
        J_train_list = np.zeros(epochs)
        J_test_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        train_num = len(train)
        if test is not None:
            test_num = len(test)
        batches_num = train_num/batch_size
        # print batches_num
        for epoch in range(epochs):
            np.random.shuffle(train)
            train_vect = np.array_split(train, batches_num)
            # print train_vect
            for batch in train_vect:
                J, dJdy = self.learn_minibatch(model, batch, loss, optimizer)
                J_train_list[epoch] += J/batches_num
                dJdy_list[epoch] += dJdy/batches_num

            if test:
                for x,t in test:
                    J_test_list[epoch] += np.linalg.norm(loss.loss(model.forward(x),t))/test_num

                if self.show_training:
                    if self.show_function is not None:
                        self.show_function(epoch, J_train_list, dJdy_list, J_test_list)
                    else:
                        print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_test_list[epoch])

            else:
                if self.show_training:
                    if self.show_function is not None:
                        self.show_function(epoch, J_train_list, dJdy_list)
                    else:
                        print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        if test:
            return J_train_list, dJdy_list, J_test_list
        return J_train_list, dJdy_list

    def on_line_learn(self, model, train, loss, optimizer, epochs, test = None):
        J_train_list = np.zeros(epochs)
        J_test_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        for epoch in range(epochs):
            np.random.shuffle(train)
            for (x,t) in train:
                J, dJdy = self.learn_one(model, x, t, loss, optimizer)
                J_train_list[epoch] += np.sum(np.power(J,2))
                dJdy_list[epoch] += np.linalg.norm(dJdy)

            if test:
                for x,t in test:
                    J_test_list[epoch] += np.sum(np.power(loss.loss(model.forward(x),t),2))

                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_test:'+str(J_test_list[epoch])

            else:
                if self.show_training:
                    print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])

        if test:
            return J_train_list, dJdy_list, J_test_list
        return J_train_list, dJdy_list
