from random import randint
import logging


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    x = randint(-5, 5)
    print(x)

    inputs = [4, -20, 34]
    weights = {0: [2, -4, 1]}
    biases = {0: 5}

    # if output negative - class 0, else class 1

def generate_random_inputs(num_inputs):
    inputs_generated = [randint(-100, 100) for i in range(len(num_inputs))] 
    return inputs_generated


if __name__ == "__main__":
    main()
