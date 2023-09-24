import logging
import math
from ex_0 import (
    get_output_non_activated,
    generate_wb_layer,
    get_network_level_output,
    get_wb_multiplied,
)

INPUTS = [4, -6, 8]
NUM_L0_NEURONS = 2
FILTER_ZERO = False
C = float("inf")


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    results, results_c = build_networks(
        inputs=INPUTS,
        num_l0=NUM_L0_NEURONS,
        constant_in=C,
    )
    logging.info(f"Results of random perceptron with default weigths :{results}")
    logging.info(
        f"Results of network with sigmoid activations and weights multiplied by constant `inf` = {C}:{results_c}"
    )
    logging.info(
        f"Are results the same for sigmoid with weights multiplied by `inf` and "
        f"perceptron activations with default weights: {(results_c[0] == results[0])}"
    )


def build_networks(inputs, num_l0, constant_in):
    len_inputs = len(inputs)
    logging.info(f"Inputs for network of perceptron = {inputs}")

    logging.info(f"Generating {num_l0} perceptron-neurons on level 0")

    weights_l0, biases_l0 = generate_wb_layer(num_inputs=len_inputs, num_neurons=num_l0)
    logging.info(f"Level 0 neurons weights values: {weights_l0}\nBiases:\n{biases_l0}")

    l0_outputs = get_network_level_output(
        inputs=inputs, weights_level=weights_l0, biases_level=biases_l0
    )
    logging.info(f"Level 0 output = {l0_outputs}")

    logging.info(f"Generating final level perceptron-neuron")
    weights_l1, biases_l1 = generate_wb_layer(num_inputs=num_l0, num_neurons=1)
    logging.info(
        f"Final level perceptron-neuron weights: {weights_l1}\nBiases:\n{biases_l1}"
    )

    l1_outputs = get_network_level_output(
        inputs=l0_outputs, weights_level=weights_l1, biases_level=biases_l1
    )

    # Constant multiptlication
    c_w_l0, c_b_l0 = get_wb_multiplied(weights_l0, biases_l0, constant_in)
    logging.info(
        f"Level 0 neurons weights values multiplied by constant C = {constant_in}: {c_w_l0}\nBiases:\n{c_b_l0}"
    )
    c_l0_outputs = get_sigmoid_neuron_output(
        inputs=inputs, weights_level=c_w_l0, biases_level=c_b_l0
    )
    logging.info(f"Level 0 sigmoid output with altered weights = {c_l0_outputs}")

    c_w_l1, c_b_l1 = get_wb_multiplied(weights_l1, biases_l1, constant_in)
    logging.info(
        f"Final level weights values multiplied by constant C = {constant_in}: {c_w_l1}\nBiases:\n{c_b_l1}"
    )

    c_l1_outputs = get_sigmoid_neuron_output(
        inputs=c_l0_outputs, weights_level=c_w_l1, biases_level=c_b_l1
    )

    return l1_outputs, c_l1_outputs


def sigmoid(z):
    activated_output = 1 / (1 + math.exp(-z))
    return activated_output


def get_sigmoid_neuron_output(inputs, weights_level, biases_level):
    assert len(weights_level.keys()) == len(biases_level.keys())
    num_neurons = len(weights_level.keys())

    level_output = []
    for i in range(num_neurons):
        non_activated_output = get_output_non_activated(
            inputs, weights_level[i], biases_level[i], filter_zero=FILTER_ZERO
        )
        sigmoid_i = sigmoid(non_activated_output)
        level_output.append(sigmoid_i)

    return level_output


if __name__ == "__main__":
    main()
