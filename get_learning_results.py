# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

from get_data import get_data
from neural_network import NeuralNetwork


def save_data(file_path: str, nn: NeuralNetwork):
    """
    Function saves data into a given file.

    param file_path: path to the file
    type: str

    param nn: NeuralNetwork instance
    type nn: NeuralNetwork
    """
    try:
        with open(file_path, "w") as file_handle:
            for line in nn.get_error():
                file_handle.write(f'{line};\n')
            print("Saving data into given file completed.")
    except IsADirectoryError:
        print("Path is a directory.")
    except FileNotFoundError:
        print("File not found.")
    except Exception:
        print("Undetermined error occurred. Please try again.")


def get_learining_results():
    """
    Function runs simulation several times in order to collect data.
    """
    # Hiperparameters
    hidden_layer = 30
    epochs = 1000
    minibatch_size = 8

    # Alternative datasets:
    # training_df, test_df = get_data('data/winequality-white.csv', ';')
    training_df, test_df = get_data('data/winequality-red.csv', ';')
    # training_df, test_df = get_data('data/iris.csv', ',')
    # training_df, test_df = get_data('data/wine-simple.csv', ';')

    # for learning_rate in (0.00015, 0.0015, 0.015, 0.15):
    for learning_rate in (0.00015, 0.0015):
        input_layer = training_df.shape[1] - 1  # Number of attributes
        # Categories are stored as an array index (0 - max_category+1)
        # therefore there are max_category+1 output neurons
        output_layer = training_df.iloc[:, -1].max() + 1

        nn = NeuralNetwork(
            input_layer, hidden_layer,
            output_layer, learning_rate
        )
        nn.train(training_df, epochs, minibatch_size)
        save_data(f'results_learning_rate_{learning_rate}.csv', nn)


if __name__ == '__main__':
    get_learining_results()
