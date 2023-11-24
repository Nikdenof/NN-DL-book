from nn.src.network import Network
import nn.src.mnist_loader as mload

import sys


def main():
    training_data, validation_data, test_data = \
    mload.load_data_wrapper()

    net = Network([2, 3, 1])
    net.feedforward([4, 3])

    sys.exit(0)


if __name__ == "__main__":
    main()
