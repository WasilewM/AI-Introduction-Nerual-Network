# import pandas as pd
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
        self.learning_rate = 0.01

    def sigmoid(self, x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def calculations_in_hidden_layer(self, input):
        # self.hidden_neurons_val = [
        #     self.sigmoid(np.dot(input, neuron))
        #     for neuron in self.input_weights
        # ]
        self.hidden_neurons_val = self.sigmoid(
            np.dot(input, self.input_weights)
        )

    def calculations_in_output_neurons(self):
        self.output_neurons_val = np.dot(
            self.hidden_neurons_val, self.output_weights
        )

    def get_mean_square_errors(self, expected_class):
        self.expected_output = [0] * self.output_neurons_num
        self.expected_output[int(expected_class)] = 1
        # self.expected_output = np.asarray(self.expected_output)
        self.errors = [
            # 0.5 * (output - error) ** 2
            output - error
            for output, error in zip(
                self.output_neurons_val, self.expected_output
            )
        ]
        self.errors = np.asarray(self.errors)

    def back_propagation(self):
        delta = [
            error * self.sigmoid(output, True)
            for error, output in zip(self.errors, self.output_neurons_val)
        ]
        for i in range(self.output_neurons_num):
            self.output_weights[i] += self.learning_rate * np.dot(
                self.hidden_neurons_val[i], delta[i]
            )

    def train(self, training_data, epochs=1):
        for _ in range(epochs):
            for row in training_data:
                # do not use class value for calculations
                class_value = row[-1]
                values = row[:-1]
                self.calculations_in_hidden_layer(values)
                self.calculations_in_output_neurons()
                self.get_mean_square_errors(class_value)
                self.back_propagation()

    def predict(self):
        pass
