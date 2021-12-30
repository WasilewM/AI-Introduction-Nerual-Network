import numpy as np


class NeuralNetwork:
    def __init__(
            self,
            input_l_c: int,
            hidden_l_c: int,
            output_l_c: int,
            learning_rate: float):
        """
        Constructor for NeuralNetwork class.

        param input_l_c: represents the number of input layer neurons
        type input_l_c: int

        param hidden_l_c: represents the number of hidden layer neurons
        type hidden_l_c: int

        param output_l_c: represents the number of output layer neurons
        type output_l_c:int

        param learning_rate: represents the learning rate of the Neural Network
        param learning_rate: float
        """
        self._lr = learning_rate
        s = 1 / np.sqrt(input_l_c)
        self._w1 = np.random.uniform(-s, s, size=(hidden_l_c, input_l_c))
        self._w2 = np.zeros((output_l_c, hidden_l_c))
        self._error = []

    def forward_propagation(self, x):
        """
        Function propagates the values forward to next Neural Network layers.

        param x: represents input layer values
        type x: numpy.ndarray
        """
        # propagate values to hidden layer
        self._z1 = self._w1 @ x
        self._a1 = sigmoid(self._z1)
        # propagate values to output layer
        self._z2 = self._w2 @ self._a1
        self._a2 = softmax(self._z2)

    def backward_propagation(self, x, y):
        """
        Function propagates the values backwards - from output layer to input
        layer of the Neural Network

        param x:
        type x: numpy.ndarray

        param y:
        type y: numpy.ndarray
        """
        m = x.shape[1]
        dz2 = self._a2 - y
        dw2 = 1/m * dz2 @ self._a1.T
        dz1 = self._w2.T @ dz2 * sigmoid(self._z1, True)
        dw1 = 1/m * dz1 @ x.T

        # update parameters
        self._w1 = self._w1 - self._lr * dw1
        self._w2 = self._w2 - self._lr * dw2

        self._last_err = np.square(dz2).mean()

    def train(self, training_df, epochs, batch_size=3):
        """
        Function manages the training process of the Neural Network.

        param training_df: training dataset
        type training_df: pandas.DataFrame

        param epochs: represents the number of epochs of the training process
        type epochs: int

        param batch_size: represents the number of data sample that should be
            taken into single batch, default value equals 3
        type batch_size: int
        """
        x = training_df.iloc[:, :-1].to_numpy().T
        y_raw = training_df.iloc[:, -1].to_numpy().T
        y = encode_y(y_raw, self._w2.shape[0])
        batches = x.T.shape[0] // batch_size
        for _ in range(epochs):
            for xs, ys in zip(
                np.array_split(x.T, batches),
                np.array_split(y.T, batches)
            ):
                self.forward_propagation(xs.T)
                self.backward_propagation(xs.T, ys.T)
            self._error.append(self._last_err)

    def predict(self, row):
        """
        Function predicts the class of the data sample.

        param row: represents data sample
        type row: numpy.ndarray
        """
        self.forward_propagation(row)
        p_category = np.argmax(self._a2, 0)
        p_probability = self._a2[p_category]
        return p_category, p_probability

    def get_error(self):
        """
        Getter for self._error attribute.
        """
        return self._error


def sigmoid(x, derivative=False):
    """
    Returns the value of the sigmoid function or its derivative ... .

    param x: represents the minibatch?
    type x: numpy.ndarray

    param derivative: answers the question if the derivative of the sigmoid
        function should be returned, default value equals False
    type derivative bool
    """
    if derivative:
        sx = sigmoid(x)
        return sx*(1-sx)
    return 1/(1+np.exp(-x))


def softmax(x):
    """
    Returns the value of the softmax function - probabilities of data class
        for all samples in the minibatch.

    param x: array representing minibatch
    type x: numpy.ndarray
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def encode_y(y, bits):
    """
    Encodes correct answers of the minibatch. Transformes values in y array
        from int in range (0,bits) to 0s and 1s, where 1 is the correct answer.

    param y: represent correct answers of the minibatch
    type y: numpy.ndarray

    param bits: represents the number? of output layer neurons - determines
        the answer
    type bits: int
    """
    encoded = np.zeros((bits, y.size), dtype=int)
    encoded[y, np.arange(y.size)] = 1
    return encoded


def get_accuracy(nn, test_df):
    """
    Returns the predictions accuracy of the Neural Network.

    param nn: neural network instance
    type nn: Neural Network

    param test_df: test dataset
    type test_df: pandas.DataFrame
    """
    correct_count = 0
    for row in test_df.to_numpy():
        m, mp = nn.predict(row[:-1])
        ex = int(row[-1])
        correct_count += 1 if m == ex else 0
        # print(f'got {m} ({mp:.2%}) expected {ex}')
    return correct_count / float(len(test_df.index))
