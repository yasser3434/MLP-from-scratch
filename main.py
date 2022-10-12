from tabnanny import verbose
import numpy
import matplotlib.pyplot as plt


def tanh(x):
    return (1.0 - numpy.exp(-2 * x)) / (1.0 + numpy.exp(-2 * x))


def tanh_derivative(x):
    return (1 + x) * (1 - x)


class NeuralNetwork:
    
    def __init__(self, net_arch):
        numpy.random.seed(0)

        # Initialized the weights
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.weights = []

        # Random initialization with range of weight values (-1,1)
        for layer in range(self.layers - 1):
            w = 2 * numpy.random.rand(net_arch[layer] + 1, net_arch[layer + 1]) - 1
            self.weights.append(w)

    def _forward_prop(self, x):
        y = x

        for i in range(len(self.weights) - 1):
            activation = numpy.dot(y[i], self.weights[i])
            activity = self.activity(activation)

            # add the bias for the next layer
            activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
            y.append(activity)

        # last layer
        activation = numpy.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)

        return y

    def _back_prop(self, y, target, learning_rate):
        error = target - y[-1]
        delta_vec = [error * self.activity_derivative(y[-1])]

        # we need to begin from the back, from the next to last layer
        for i in range(self.layers - 2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error * self.activity_derivative(y[i][1:])
            delta_vec.append(error)

        # Now we need to set the values from back to front
        delta_vec.reverse()

        # Finally, we adjust the weights, using the backpropagation 
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.arch[i] + 1)
            delta = delta_vec[i].reshape(1, self.arch[i + 1])
            self.weights[i] += learning_rate * layer.T.dot(delta)

 
    def fit(self, data, labels, learning_rate=0.1, epochs=100):

        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)

        for k in range(epochs):
            if (k + 1) % 10000 == 0:
                print('epochs: {}'.format(k + 1))

            sample = numpy.random.randint(data.shape[0])

            # We will now go ahead and set up our feed-forward propagation:
            x = [Z[sample]]
            y = self._forward_prop(x)

            # Now we do our back-propagation of the error to adjust the weights:
            target = labels[sample]
            self._back_prop(y, target, learning_rate)

 
    def predict_single_data(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return val[1]

    def predict(self, X):
        Y = numpy.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            Y = numpy.vstack((Y, y))
        return Y



if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1])

    # Set the input data
    X = numpy.array([[0.1, 0.1], [0.1, 1.1],
                     [1.9, 0.1], [1.8, 1.9]])

    # Set the labels, the correct results for the xor operation
    y = numpy.array([-1, 1,
                     1, -1])

    print("X Shape :", X.shape)
    print("y Shape :", y.shape)

    # Call the fit function and train the network for a chosen number of epochs
    nn.fit(X, y, epochs=10000)

    # Show the prediction results
    print("Final prediction")
    # for s in X:
    #     print(s, nn.predict_single_data(s))

    print(nn.predict(X))
    