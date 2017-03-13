import matplotlib.pyplot as plt
import numpy as np

class Printer2D():
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

    def compare_data(self, figure_ind, data, targets, output, num_classes, classes):
        x = range(num_classes)
        y = range(num_classes)
        x_err = range(num_classes)
        y_err = range(num_classes)

        for type in range(num_classes):
            x[type] = []
            y[type] = []
            x_err[type] = []
            y_err[type] = []

        for type in range(num_classes):
            for i, point in enumerate(data):
                if np.argmax(targets[i]) == np.argmax(output[i]):
                    x[np.argmax(targets[i])].append(point[0])
                    y[np.argmax(targets[i])].append(point[1])
                else:
                    x_err[np.argmax(targets[i])].append(point[0])
                    y_err[np.argmax(targets[i])].append(point[1])

        plt.figure(figure_ind)
        for type in range(num_classes):
            plt.scatter(x[type], y[type], s=100, color='green', marker=classes[type])
            plt.scatter(x_err[type], y_err[type], s=100, color='red', marker=classes[type])

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
