import matplotlib.pyplot as plt
import numpy as np

class Printer2D():
    def forward_all(self, model, xs):
        return map(lambda x: model.forward(x), xs)

    def draw_decision_surface(self, figure_ind, model, data):
        max = np.max([i[0] for i in data],0)
        min = np.min([i[0] for i in data],0)
        x_range = np.linspace(min[0]-0.5,max[0]+0.5,100)
        y_range = np.linspace(min[1]-0.5,max[1]+0.5,100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.argmax(self.forward_all(model, np.c_[X.ravel(), Y.ravel()]), axis=1)
        Z = Z.reshape(X.shape)
        # cs = plt.contourf(xx, yy, Z, cmap='Paired')
        plt.figure(figure_ind)
        plt.set_cmap(plt.cm.Paired)
        plt.pcolormesh(X, Y, Z)

    def print_model(self, figure_ind, model, x_list):
        max = np.max(x_list,0)
        min = np.min(x_list,0)
        x_range = np.linspace(min[0],max[0],100)
        y_range = np.linspace(min[1],max[1],100)
        X, Y = np.meshgrid(x_range, y_range)
        for ind,layer in enumerate(model.layers):
            plt.figure(figure_ind+ind)
            z_vett = []
            for x_ind in xrange(X.shape[0]):
                for y_ind in xrange(Y.shape[1]):
                    aux_x = np.array([X[x_ind][y_ind],Y[x_ind][y_ind]])
                    for layer in model.layers[:ind]:
                        aux_x = layer.forward(aux_x)
                    z_vett.append(aux_x)

            z_array = np.array(z_vett)
            for exit in range(z_array.shape[1]):
                z_array_out = z_array[:,exit]
                Z = z_array_out.reshape(X.shape)
                plt.subplot(z_array.shape[1], 1, exit+1)
                try:
                    CS = plt.contour(X, Y, Z)
                except ValueError:
                    pass

    def compare_data(self, figure_ind, train, output, num_classes, colors, classes):
        xy = range(num_classes)
        for type in range(num_classes):
            xy[type] = []

        for type in range(num_classes):
            for i, (x,t) in enumerate(train):
                xy[np.argmax(t)].append((x[0],x[1],np.argmax(output[i])))

        plt.figure(figure_ind)
        for type in range(num_classes):
            for x,y,c in xy[type]:
                plt.scatter(x, y, s=100, color=colors[c], marker=classes[type])

    def print_data(self, figure_ind, data, targets, num_classes, colors, classes):
        x = range(num_classes)
        y = range(num_classes)
        for type in range(num_classes):
            x[type] = [point[0] for i, point in enumerate(data) if np.argmax(targets[i]) == type]
            y[type] = [point[1] for i, point in enumerate(data) if np.argmax(targets[i]) == type]

        plt.figure(figure_ind)
        for type in range(num_classes):
            plt.scatter(x[type], y[type], s=100, color=colors[type], marker=classes[type])

    def to_one_hot_vect(self, vect, num_classes):
        on_hot_vect = []
        for i,target in enumerate(vect):
            on_hot_vect.append(np.zeros(num_classes))
            on_hot_vect[i][target] = 1
        return on_hot_vect
