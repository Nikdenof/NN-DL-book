import logging
from math import log
from random import randint


# todo: make len_inputs random
LEN_INPUTS = 3
NUM_L0_NEURONS = 2


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    results = build_perceptron_network(
            len_inputs=LEN_INPUTS, 
            num_l0=NUM_L0_NEURONS
            )
    logging.info(f"Results of random perceptron with default weigths {results}")


def build_perceptron_network(len_inputs, num_l0):
    logging.info(f"Generating {len_inputs} inputs")

    inputs = generate_random_lst(len_inputs)
    logging.info(f"Inputs for network of perceptron = {inputs}")

    logging.info(f"Generating {num_l0} perceptron-neurons on level 0")

    weights_l0, biases_l0 = generate_wb_layer(num_inputs=len_inputs, num_neurons=num_l0)  
    logging.info(f"Level 0 neurons weights values: {weights_l0}\nBiases:\n{biases_l0}")

    l0_outputs = get_network_level_output(inputs=inputs, weights_level=weights_l0, biases_level=biases_l0)
    logging.info(f"Level 0 output = {l0_outputs}")

    logging.info(f"Generating final level perceptron-neuron")
    weights_l1, biases_l1 = generate_wb_layer(num_inputs=num_l0, num_neurons=1) 
    logging.info(f"Final level perceptron-neuron weights: {weights_l1}\nBiases:\n{biases_l1}")

    l1_outputs = get_network_level_output(inputs=l0_outputs, weights_level=weights_l1, biases_level=biases_l1)

    return l1_outputs


def get_perceptron_output(inputs, weights, bias):
    assert len(inputs) == len(weights)

    result = 0  
    for i in range(len(inputs)):
        result += inputs[i] * weights[i]
    logging.debug(f"Results before applying bias {result}")

    result += bias
    logging.debug(f"Results before thresholding {result}")

    # if output positive - class 1, else class 0
    if result > 0:
        return 1
    else:
        return 0


def get_network_level_output(inputs, weights_level, biases_level):
    assert len(weights_level.keys()) == len(biases_level.keys())
    num_neurons = len(weights_level.keys())  

    level_output = []
    for i in range(num_neurons):
        perceptron_output_i = get_perceptron_output(
                inputs=inputs, 
                weights=weights_level[i], 
                bias=biases_level[i]
                )
        level_output.append(perceptron_output_i)

    return level_output


def generate_random_lst(len_lst):
    generated_lst = [randint(-100, 100) for _ in range(len_lst)] 
    return generated_lst 


def generate_wb_layer(num_inputs, num_neurons):
    weights_dict = {}
    biases_dict = {}
    for i in range(num_neurons):
        weights_dict[i] = generate_random_lst(num_inputs) 
        biases_dict[i] = randint(-1000, 1000)
    return weights_dict, biases_dict


if __name__ == "__main__":
    main()
