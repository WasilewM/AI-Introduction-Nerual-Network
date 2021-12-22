# import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from get_data import get_data


def run():
    data = get_data()
    data = np.asarray(data)
    training_set = data[:900]
    test_set = data[900:]
    # print(data)
    # data = pd.read_csv('winequality-red.csv')
    nn = NeuralNetwork()
    nn.train(training_set)
    # print(np.asarray(nn.input_weights))
    # print(np.asarray(nn.output_weights))
    sample = test_set[:1]
    print(nn.predict(sample[0][:-1]))
    print(test_set[:1][0][-1])


if __name__ == "__main__":
    run()
