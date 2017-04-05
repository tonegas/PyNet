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
        return J, dJdy

    def learn(self, model, train, loss, optimizer, epochs, batch_size = 1, validation = None):
        J_train_list = np.zeros(epochs)
        J_validation_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        train_num = len(train)
        if validation is not None:
            validation_num = len(validation)
        batches_num = train_num/batch_size
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

                    print y,t
                    dJdy = loss.dJdy_gradient(y,t)

                    if this_batch_size == i+1:
                        optimizer.store = False

                    model.backward(dJdy, optimizer)

                    J_train_list[epoch] += np.linalg.norm(J)/train_num
                    dJdy_list[epoch] += np.linalg.norm(dJdy)/train_num


            if validation:
                for x,t in validation:
                    J_validation_list[epoch] += np.linalg.norm(loss.loss(model.forward(x),t))/validation_num

                if self.show_training:
                    if self.show_function is not None:
                        self.show_function(epoch, J_train_list, dJdy_list, J_validation_list)
                    else:
                        print 'Epoch:'+str(epoch)+' J_train:'+str(J_train_list[epoch])+' J_validation:'+str(J_validation_list[epoch])

            else:
                if self.show_training:
                    if self.show_function is not None:
                        self.show_function(epoch, J_train_list, dJdy_list)
                    else:
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
