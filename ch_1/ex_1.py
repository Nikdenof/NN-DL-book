import logging


LEN_INPUTS = 3
NUM_L0_NEURONS = 2
C = 3


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)


if __name__ == "__main__":
    main()
