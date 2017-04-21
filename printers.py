import matplotlib.pyplot as plt
import numpy as np

class ShowTraining():
    def __init__(self, epochs_num = None):
        plt.ion()
        self.fig1 = plt.figure(1)
        self.ax = self.fig1.add_subplot(111)
        self.ax.set_title('Errors History (J)')
        self.train, = self.ax.plot(xrange(len([])), [], color='green', marker='^', label='Training')
        self.test, = self.ax.plot(xrange(len([])), [], color='blue', marker='s', label='Test')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel(r'$||J||_2/N$')
        self.ax.legend()
        if epochs_num is not None:
            self.ax.set_xlim([0,epochs_num])

        self.fig2 = plt.figure(2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title('Loss Gradient History (dJ/dy)')
        self.dJdy, = self.ax2.plot(xrange(len([])), [], color='red', marker='o')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel(r'$||\delta J/\delta y||_2/N$')
        if epochs_num is not None:
            self.ax2.set_xlim([0,epochs_num])

    def show(self, epoch, J_train_list, dJdy_list = None, J_test_list = None):
        self.train.set_xdata(xrange(len(J_train_list[:epoch+1])))
        self.train.set_ydata(J_train_list[:epoch+1])
        if J_test_list is not None:
            self.test.set_xdata(xrange(len(J_test_list[:epoch+1])))
            self.test.set_ydata(J_test_list[:epoch+1])
            self.ax.set_ylim([0,max(max(J_train_list)+max(J_train_list)*0.1,max(J_test_list)+max(J_test_list)*0.1)])
        else:
            self.ax.set_ylim([0,max(J_train_list)+max(J_train_list)*0.1])
        self.fig1.canvas.draw()

        self.dJdy.set_xdata(xrange(len(dJdy_list[:epoch+1])))
        self.dJdy.set_ydata(dJdy_list[:epoch+1])
        self.ax2.set_ylim([min(dJdy_list)-min(dJdy_list)*0.1,max(dJdy_list)+max(dJdy_list)*0.1])
        self.fig2.canvas.draw()


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
        for ind,layer in enumerate(model.elements):
            plt.figure(figure_ind+ind)
            z_vett = []
            for x_ind in xrange(X.shape[0]):
                for y_ind in xrange(Y.shape[1]):
                    aux_x = np.array([X[x_ind][y_ind],Y[x_ind][y_ind]])
                    for layer in model.elements[:ind]:
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

