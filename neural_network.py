import numpy as np


class NeuralNetwork:
    def __init__(
            self,
            input_l_c: int,
            hidden_l_c: int,
            output_l_c: int,
            learning_rate: float):
        self._lr = learning_rate
        s = 1 / np.sqrt(input_l_c)
        self._w1 = np.random.uniform(-s, s, size=(hidden_l_c, input_l_c))
        self._w2 = np.zeros((output_l_c, hidden_l_c))
        self._error = []

    def forward_propagation(self, x):
        self._z1 = self._w1 @ x
        self._a1 = sigmoid(self._z1)
        self._z2 = self._w2 @ self._a1
        self._a2 = softmax(self._z2)

    def backward_propagation(self, x, y):
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
        x = training_df.iloc[:, :-1].to_numpy().T
        y_raw = training_df.iloc[:, -1].to_numpy().T
        y = encode_y(y_raw, self._w2.shape[0])
        batches = x.T.shape[0] // batch_size
        for _ in range(epochs):
            for xs, ys in zip(np.array_split(x.T, batches), np.array_split(y.T, batches)):
                self.forward_propagation(xs.T)
                self.backward_propagation(xs.T, ys.T)
            self._error.append(self._last_err)

    def predict(self, row):
        self.forward_propagation(row)
        p_category = np.argmax(self._a2, 0)
        p_probability = self._a2[p_category]
        return p_category, p_probability

    def get_error(self):
        return self._error


def sigmoid(x, derivative=False):
    if derivative:
        sx = sigmoid(x)
        return sx*(1-sx)
    return 1/(1+np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def encode_y(y, bits):
    encoded = np.zeros((bits, y.size), dtype=int)
    encoded[y, np.arange(y.size)] = 1
    return encoded


def get_accuracy(nn, test_df):
    correct_count = 0
    for row in test_df.to_numpy():
        m, mp = nn.predict(row[:-1])
        ex = int(row[-1])
        correct_count += 1 if m == ex else 0
        # print(f'got {m} ({mp:.2%}) expected {ex}')
    return correct_count / float(len(test_df.index))
