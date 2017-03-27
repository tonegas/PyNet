# PyNet
PyNet is a easy and didactic framework for machine learning with neural networks.
The focus of the framework is understanding in a easy way the basic concepts of a a neural network thanks to a short and clear implementation in python.
In the framework are implemented some basic networks as Hopfield network, Kohonen network, and it is easy to implement every layers networks.

# The Architecture
The architecture of PyNet is pretty similar to Torch7 and Keras interfaces.
The main modules of the framework are:
- __genericlayer.py__ It is the main common classes.
- layers.py. All the main classic layers already implemented as:
    - LinearLayer
    - SoftMaxLayer
    - SigmoidLayer
    - TanhLayer
    - etc...
- network.py. This file contains the main structure for create a computational graph as:
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
    These two classes manage of a vector input and generate a vector output.
    Moreover a structure to manage group of vector, to easy create a computational graph as:
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
- losses.py. In this file there are the losses function as:
    - SquareLoss
    - NegativeLogLikelihoodLoss
    - CrossEntropyLoss

- optimizer.py. Here there are the main optimizer classes as:
    - GradientDescent
    - GradientDescentMomentum

- trainer.py. In this file there is the class for the training.

## Complete Example of classification
```python

    train = #GET TRAIN DATA

    from layers import LinearLayer, SoftMaxLayer, SigmoidLayer
    from losses import NegativeLogLikelihoodLoss
    from optimizers import GradientDescentMomentum
    from network import Sequential
    from trainer import Trainer
    from printers import Printer2D, ShowTraining

    model = Sequential(
        LinearLayer(2, 3),
        SigmoidLayer,
        LinearLayer(3, 4),
        SoftMaxLayer,
    )

    display = ShowTraining(epochs_num = 5)
    trainer = Trainer(show_training = True, show_function = display.show)

    J_train_list, dJdy_list = trainer.learn(
        model = model,
        train = train,
        loss = NegativeLogLikelihoodLoss(),
        optimizer = GradientDescentMomentum(learning_rate=0.011, momentum=0.1),
        batch_size = 10,
        epochs = 5
    )
```
## Create a NewLayer


