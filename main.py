import matplotlib.pyplot as plt
import numpy as np
import pathlib


from get_data import get_data
from neural_network import NeuralNetwork, get_accuracy


def plot_error(nn: NeuralNetwork, title: str, filename: str):
    """
    Function plots the chart of the error of the Neural Network at the end of
    each learning epoch.

    param nn: neural network instance
    type nn: NeuralNetwork

    param filename: path where plot will be saved
    type filename: str
    """
    error = nn.get_error()
    y_values = error
    x_values = range(1, len(error)+1)
    plt.style.use('dark_background')
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(filename)
    plt.close()


def plot_stats_combined(stats_combined, title, filename):
    """
    Function plots the chart of the error of the Neural Network at the end of
    each learning epoch. (6 combined)

    param stats_combined: neural network instance
    type nn: NeuralNetwork

    param filename: path where plot will be saved
    type filename: str
    """
    plt.style.use('dark_background')
    fig, _ = plt.subplots(3, 2)
    fig.set_size_inches(12, 8)
    fig.set_dpi(100)
    fig.suptitle(title)
    hr = np.maximum.reduce([stats['error'] for stats in stats_combined]).max()
    lr = np.minimum.reduce([stats['error'] for stats in stats_combined]).min()
    for ax, stats in zip(fig.get_axes(), stats_combined):
        subtitle = r'$\alpha$ = ' + \
            f"{stats['lr']}, accuracy: {stats['accuracy']:.2%}"
        ax.set_ylim([lr * 0.8, hr * 1.2])
        y_values = stats['error']
        x_values = range(1, len(y_values)+1)
        ax.plot(x_values, y_values, ':.')
        ax.set_title(subtitle, fontsize=10)
        ax.label_outer()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def main():
    """
    Main function f the simulation.
    """
    np.random.seed(0)

    #
    # Hiperparameters (Red Wine)
    #
    learning_rates = [0.0001, 0.0005, 0.006, 0.0105, 0.08, 0.2]
    hidden_layer = 11
    epochs = 800
    minibatch_size = 8

    training_df, test_df = get_data('data/winequality-red.csv', ';')
    # Alternative datasets:
    # training_df, test_df = get_data('data/winequality-white.csv', ';')
    # training_df, test_df = get_data('data/iris.csv', ',')
    # training_df, test_df = get_data('data/wine-simple.csv', ';')

    input_layer = training_df.shape[1] - 1  # Number of attributes
    # Categories are stored as an array index (0 - max_category+1)
    # therefore there are max_category+1 output neurons
    output_layer = training_df.iloc[:, -1].max() + 1

    stats_combined = []

    for lr in learning_rates:
        np.random.seed(0)
        nn = NeuralNetwork(input_layer, hidden_layer, output_layer, lr)
        nn.train(training_df, epochs, minibatch_size)
        accuracy = get_accuracy(nn, test_df)
        print(f'accuracy (Red wine, alpha={lr}):\t{accuracy:.2%}')
        stats_combined.append({
            'error': nn.get_error(),
            'lr': lr,
            'hl': hidden_layer,
            'epochs': epochs,
            'batch': minibatch_size,
            'accuracy': accuracy
        })

    pathlib.Path('img').mkdir(parents=True, exist_ok=True)
    plot_stats_combined(stats_combined, 'Red Wine', 'img/red_wine.png')

    #
    # Hiperparameters (Iris)
    #
    learning_rate = 0.01
    hidden_layer = 8
    epochs = 1000
    minibatch_size = 8
    training_df, test_df = get_data('data/iris.csv', ',')
    input_layer = training_df.shape[1] - 1
    output_layer = training_df.iloc[:, -1].max() + 1
    nn = NeuralNetwork(input_layer, hidden_layer, output_layer, learning_rate)
    nn.train(training_df, epochs, minibatch_size)
    accuracy = get_accuracy(nn, test_df)
    print(f'accuracy (Iris, alpha={learning_rate}):\t\t{accuracy:.2%}')
    plot_error(nn, f'Iris (accuracy: {accuracy:.2%})', 'img/iris.png')


if __name__ == '__main__':
    main()
