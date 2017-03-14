import numpy as np
from itertools import izip

class Trainer():
    def __init__(self, depth = 1, show_training = False):
        self.depth = depth
        self.show_training = show_training

    def train(self, model, dJdy, optimizer, depth = None):
        if depth == None:
            depth = self.depth

        aux_dJdy = dJdy
        if model.group:
            if hasattr(model,'layers'):
                for layer in reversed(model.layers):
                    # print 'dJdy'+str(aux_dJdy)
                    aux_dJdx = layer.backward(aux_dJdy)
                    #print aux_dJdy
                    if hasattr(layer,'dJdW_gradient') and hasattr(optimizer, 'update'):
                        # print 'grad'+str(np.max(layer.dJdW_gradient(aux_dJdy)))
                        optimizer.update(layer, layer.dJdW_gradient(aux_dJdy))
                    elif depth-1 > 0:
                        self.train(layer, aux_dJdy, optimizer, depth-1)
                    aux_dJdy = aux_dJdx
            else:
                aux_dJdx = []
                a = 0
                for ind,element in enumerate(model.elements):
                    b = a+model.vect_size[ind]
                    aux_dJdx.append(element.backward(aux_dJdy[a:b]))
                    if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                        optimizer.update(element, element.dJdW_gradient(aux_dJdy[a:b]))
                    elif depth-1 > 0:
                        self.train(element, aux_dJdy[a:b], optimizer, depth-1)
                    a = b
                aux_dJdy = np.sum(np.array(aux_dJdx),0)
        else:
            aux_dJdx = model.backward(aux_dJdy)
            if hasattr(model,'dJdW_gradient') and hasattr(optimizer, 'update'):
                optimizer.update(model, model.dJdW_gradient(aux_dJdy))
            aux_dJdy = aux_dJdx

        return aux_dJdy

    def learn_one(self, model, input, target, loss, optimizer):
        y = model.forward(input)
        J = loss.loss(y,target)
        dJdy = loss.dJdy_gradient(y,target)
        self.train(model, dJdy, optimizer)
        return J, dJdy

    def learn(self, model, input_data, target_data, loss, optimizer, epochs):
        J_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        num_sample = len(input_data)
        for epoch in range(epochs):
            p = np.random.permutation(len(input_data))
            input_data = input_data[p]
            target_data = target_data[p]
            for x,t in izip(input_data,target_data):
                J, dJdy = self.learn_one(model, x, t, loss, optimizer)
                J_list[epoch] += np.sqrt(np.sum(np.power(J,2)))/num_sample
                dJdy_list[epoch] += np.mean(dJdy)/num_sample

            if self.show_training:
                print 'Epoch:'+str(epoch)+' J:'+str(J_list[epoch])

        return J_list, dJdy_list

    #Qui c'e' il bug dovuto al fatto che non si puo'
    def learn_minibatch(self, model, input_data, target_data, loss, optimizer, epochs, batches_num):
        J_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        num_sample = len(input_data)
        for epoch in range(epochs):
            p = np.random.permutation(len(input_data))
            input_data_vect = np.array_split(input_data[p],batches_num)
            target_data_vect = np.array_split(target_data[p],batches_num)
            for batch in range(batches_num):
                mean_dJdy = np.zeros(target_data[0].shape)
                mean_x = np.zeros(input_data[0].shape)
                dim_input_data = len(input_data_vect[batch])
                for x,t in izip(input_data_vect[batch],target_data_vect[batch]):
                    # print 'x'+str(x)
                    y = model.forward(x)
                    # print 'y'+str(y)
                    J = loss.loss(y,t)

                    dJdy = loss.dJdy_gradient(y,t)

                    mean_x += x/float(dim_input_data)
                    mean_dJdy += dJdy/float(dim_input_data)
                    # print dJdy
                    J_list[epoch] += np.sqrt(np.sum(np.power(J,2)))/float(num_sample)
                    dJdy_list[epoch] += np.mean(dJdy)/float(num_sample)

                y = model.forward(mean_x)
                self.train(model, mean_dJdy, optimizer)

            if self.show_training:
                print 'Epoch: '+str(epoch)+' J: '+str(J_list[epoch])

        return J_list, dJdy_list
#