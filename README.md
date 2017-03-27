# PyNet
PyNet is a easy and didactic framework for machine learning with neural networks.
The focus of the framework is understanding in a easy way the basic concepts of a a neural network thanks to a short and clear implementation in python.
In the framework are implemented some basic networks as _Hopfield_ network, _Kohonen_ network, and it is easy to implement every layers networks.

# The Architecture
The architecture of PyNet is pretty similar to Torch7 and Keras interfaces.
The main modules of the framework are:
- __genericlayer.py.__ It is the main common classes.
- __layers.py.__ All the main classic layers already implemented as:
    - LinearLayer
    - SoftMaxLayer
    - SigmoidLayer
    - TanhLayer
    - etc...
- __network.py.__ This file contains the main structure for create a computational graph as:
    - Sequential: each element in sequential has to have the number of inputs equal
    to the number of outputs of the previous layer. The input is forword across the chain in the sequential elements.
    - Parallel: each element in parallel has to have the same number of inputs,
    the inputs is forwarded to each element of in the parallel.

    An Example:
    ```python
    model = Sequential(
        LinearLayer(2, 5),
        SigmoidLayer,
        LinearLayer(5, 3),
        SoftMaxLayer
    )
    #is the same of
    model = Sequential(
        LinearLayer(2, 5),
        SigmoidLayer,
        Parallel(
            LinearLayer(5, 1),
            LinearLayer(5, 1),
            LinearLayer(5, 1),
        )
        SoftMaxLayer
    )
    ```
    These two classes manage a vector input and generate a vector output.
    Moreover there are some structures to manage group of vector, in order to create a computational graph as:
    - SumGroup
    - MulGroup
    - ParallelGroup
    - MapGroup

    An Example:

    ```python
    #y = a*x^2+b*x+c
    n = Sequential(
            ParallelGroup(
                Sequential(
                    ParallelGroup(
                        GenericLayer,  # x
                        GenericLayer,  # x
                        WeightLayer(1) # a
                    ),MulLayer # x*x*a
                ),
                Sequential(
                    ParallelGroup(
                        GenericLayer,   # x
                        WeightLayer(1), # b
                    ),MulLayer # b*x
                ),
                WeightLayer(1) # constant c
            ),SumLayer # sum of all terms
        )
    #is the same of
    n2 = Sequential(
            ParallelGroup(
                Sequential(
                    ParallelGroup(
                        GenericLayer,  # x
                        GenericLayer,  # x
                        WeightLayer(1) # a
                    ),MulLayer # x*x*a
                ),
                LinearLayer(1,1) # b*x+c
            ),SumLayer # sum of all terms
        )
    #is the same of
    n = Sequential(
            ParallelGroup(GenericLayer,GenericLayer,GenericLayer), # create a vector of [x,x,x]
            SumGroup( # the vector is used in:
                Sequential( # x
                    ParallelGroup(GenericLayer,GenericLayer,GenericLayer), # create a vector of [x,x,x]
                    MulGroup(
                        GenericLayer, # x
                        GenericLayer, # x
                        WeightLayer(1) # a
                    ) # x^2*a
                ),
                Sequential( # x
                    ParallelGroup(GenericLayer,GenericLayer), # create a vector [x,x]
                    MulGroup(
                        GenericLayer, # x
                        WeightLayer(1) # b
                    ) # x*b
                ),
                WeightLayer(1) # c
            )
        )
    ```
- __losses.py.__ In this file there are the losses function as:
    - SquareLoss
    - NegativeLogLikelihoodLoss
    - CrossEntropyLoss

- __optimizer.py.__ Here there are the main optimizer classes as:
    - GradientDescent
    - GradientDescentMomentum

- __trainer.py.__ In this file there is the class for the training.

## Complete Example of classification
```python

    train = #GET TRAIN DATA

    from layers import LinearLayer, SoftMaxLayer, SigmoidLayer
    from losses import NegativeLogLikelihoodLoss
    from optimizers import GradientDescentMomentum
    from network import Sequential
    from trainer import Trainer
    from printers import Printer2D, ShowTraining

    num_classes = 4
    classes = ["o","^","x","s"]
    colors = ['r', 'g', 'b', 'y']

    model = Sequential(
        LinearLayer(2, 3),
        SigmoidLayer,
        LinearLayer(3, num_classes),
        SoftMaxLayer,
    )

    display = ShowTraining(epochs_num = 5)
    trainer = Trainer(show_training = True, show_function = display.show)

    J_train_list, dJdy_list = trainer.learn(
        model = model,
        train = train,
        loss = NegativeLogLikelihoodLoss(),
        optimizer = GradientDescentMomentum(learning_rate=0.1, momentum=0.1),
        batch_size = 10,
        epochs = 5
    )

    for i, (x,target) in enumerate(train):
        y.append(model.forward(x))

    p = Printer2D() # A class just for the 2D input of classification
    p.draw_decision_surface(1, model, train)
    p.compare_data(1, train, y, num_classes, colors, classes)
    plt.title('After Training')
```
# Why

The only good way to understand modern neural network is to try to implement one from scratch.
After that you can use TensorFlow.

