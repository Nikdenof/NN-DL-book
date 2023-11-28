from nn.src.network import Network
import nn.src.mnist_loader as mload

import sys


def main():
    training_data, validation_data, test_data = \
    mload.load_data()

    net = Network([784, 30, 10])
    print(len(training_data))

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    sys.exit(0)


if __name__ == "__main__":
    main()
