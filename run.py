# import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from get_data import get_data


def run():
    data = get_data()
    data = np.asarray(data[:1])
    # print(data)
    # data = pd.read_csv('winequality-red.csv')
    nn = NeuralNetwork()
    nn.train(data)
    print(nn.input_weights)
    print(nn.output_weights)


if __name__ == "__main__":
    run()
