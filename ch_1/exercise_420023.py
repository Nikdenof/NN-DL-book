"""
URL - (http://neuralnetworksanddeeplearning.com/chap1.html#exercise_420023)
Description:
    Try creating a network with just two layers - an input and an output layer, 
    no hidden layer - with 784 and 10 neurons, respectively. 
    Train the network using stochastic gradient descent. 
    What classification accuracy can you achieve?
"""

from nn.src.network import Network
import nn.src.mnist_loader as mload

import sys


def main():
    training_data, validation_data, test_data = \
    mload.load_data_wrapper()

    net = Network([784, 10])

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    sys.exit(0)


if __name__ == "__main__":
    main()
