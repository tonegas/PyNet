## PyNet
PyNet is a easy and didactic framework for machine learning with neural networks.
The focus of the framework is understanding in a easy way the basic concepts of a a neural network thanks to a short and clear implementation in python.
In the framework are implemented some basic networks as Hopfield network, Kohonen network, and it is easy to implement every layers networks.

The architecture of PyNet is pretty similar to Torch7 and Keras interfaces.

The main modules of the framework are:
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
