import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from get_data import get_data
from neural_network import NeuralNetwork, get_accuracy


def plot_error(nn):
    error = nn.get_error()
    y_values = error
    x_values = range(1, len(error)+1)
    plt.style.use('dark_background')
    plt.plot(x_values, y_values)
    plt.title('Neural Network')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()


def main():
    np.random.seed(0)

    # Hiperparameters
    learning_rate = 0.0015
    hidden_layer = 30
    epochs = 1000
    minibatch_size = 8

    # Alternative datasets:
    # training_df, test_df = get_data('data/winequality-white.csv', ';')
    training_df, test_df = get_data('data/winequality-red.csv', ';')
    # training_df, test_df = get_data('data/iris.csv', ',')
    # training_df, test_df = get_data('data/wine-simple.csv', ';')

    input_layer = training_df.shape[1] - 1  # Number of attributes
    # Categories are stored as an array index (0 - max_category+1)
    # therefore there are max_category+1 output neurons
    output_layer = training_df.iloc[:, -1].max() + 1

    nn = NeuralNetwork(input_layer, hidden_layer, output_layer, learning_rate)
    nn.train(training_df, epochs, minibatch_size)

    accuracy = get_accuracy(nn, test_df)
    print(f'accuracy: {accuracy:.2%}')

    plot_error(nn)


if __name__ == '__main__':
    main()
