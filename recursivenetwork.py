import genericlayer

class NodeGenerator():
    def __init__(self, node, *args, **kwargs):
        self.node = node
        self.args = args
        self.kwargs = kwargs

class RNN(genericlayer.GenericLayer, NodeGenerator):
    def __init__(self, Node, *args, **kwargs):
        NodeGenerator.__init__(self, Node, *args, **kwargs)
        self.window_size = 0
        self.window_step = 0
        self.nodes = []
        self.net = self.node(*self.args, **self.kwargs)
        self.message_fun = {
            'delete_nodes' : self.delete_nodes,
            'init_nodes' : self.init_nodes,
            'clear_memory' : self.clear_memory
        }

    def on_message(self,message,*args,**kwargs):
        self.message_fun[message](*args,**kwargs)

    def delete_nodes(self):
        self.nodes = []

    def init_nodes(self, window_size):
        self.window_size = window_size
        self.window_step = 0
        self.nodes = []
        for ind in range(window_size):
            self.nodes.append(self.node(*self.args, **self.kwargs))

    def clear_memory(self):
        if type(self.net.state) is list:
            for e in range(len(self.net.state)):
                self.net.state[e].fill(0.0)
                for ind in range(self.window_size):
                    self.nodes[ind].state[e].fill(0.0)
                    self.nodes[ind].dJdstate[e].fill(0.0)
        else:
            self.net.state.fill(0.0)
            for ind in range(self.window_size):
                self.nodes[ind].state.fill(0.0)
                self.nodes[ind].dJdstate.fill(0.0)


    def forward(self, x, update = False):
        if update:
            if self.window_step == self.window_size:
                raise Exception('Window Exceeded')
            [y, self.nodes[self.window_step].state] = self.nodes[self.window_step].forward([x, self.nodes[self.window_step - 1].state])
            self.window_step += 1
            return y
        else:
            [y,self.net.state] = self.net.forward([x, self.net.state])
            return y

    def backward(self, dJdy, optimizer = None):
        if self.window_step == 0:
            raise Exception('Window Exceeded')
        self.window_step -= 1
        [dJdx, dJdh] = self.nodes[self.window_step].backward([dJdy, self.nodes[self.window_step].dJdstate], optimizer)
        if self.window_step > 0:
            self.nodes[self.window_step-1].dJdstate = dJdh
        return dJdx
