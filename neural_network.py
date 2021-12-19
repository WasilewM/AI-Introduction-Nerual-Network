import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, input_neurons=11, hidden_neurons=11, output_neurons=11):
        self.input_neurons_num = input_neurons
        self.hidden_neurons_num = hidden_neurons
        self.output_neurons_num = output_neurons
        self.input_weights = [
            [0.5] * self.input_neurons_num
            for _ in range(self.hidden_neurons_num)
        ]
        self.output_weights = [
            [0.5] * self.output_neurons_num
            for _ in range(self.hidden_neurons_num)
        ]

    def activation_func(self, x):
        return 1 / (1 + np.exp(-x))

    def back_propagation(self):
        pass

    def calculations_in_hidden_layer(self, input):
        self.hidden_neurons_val = [
            self.activation_func(np.dot(input, neuron))
            for neuron in self.input_weights
        ]

    def calculations_in_output_neurons(self):
        # self.output_neurons_val = [
        #     np.dot(hidden, output)
        #     for hidden, output in zip(self.hidden_neurons_val, self.output_weights)
        # ]
        self.output_neurons_val = [
            np.dot(self.hidden_neurons_val, output)
            for output in self.output_weights
        ]

    def train(self, training_data, epochs=1):
        for _ in range(epochs):
            for row in training_data:
                # do not use class value for calculations
                class_value = row[-1]
                values = row[:-1]
                self.calculations_in_hidden_layer(values)
                self.calculations_in_output_neurons()

    def predict(self):
        pass
