import time

import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import matplotlib.pyplot as plt
import numpy as np


# the ReLu function
def relu(x):
    return np.maximum(x, 0)


# the derivative of the ReLu function
def deriv_relu(x):
    return np.maximum(np.sign(x), 0)


# the softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


ACTIVATION_FUNCTIONS = {'relu': relu, 'softmax': softmax}
DERIV_ACTIVATION_FUNCTION = {'relu': deriv_relu, 'softmax': deriv_relu}


class MLPmodel:
    """This is a from-scratch implementation of a multi-layer perceptron."""

    def __init__(self, verbose=False, plc=False):
        self._input_dim = -1  # the dimension of the input vector
        self.layers = []  # list of all the network's hidden layers
        self.activation_func = []  # list of all the activation function for each layer and their derivatives
        self.deriv_activation_func = []
        self.bias = []  # List of the biases for each layer
        self.one_hot = LabelBinarizer()  # one hot encoder for the output classes
        self._verbose = verbose  # if verbose is True there will be more information printed during the fitting process
        self._plot_learning_curve = plc  # if True the classifier will plot the learning curve after the training process

    def add_layer(self, layer_size: int, input_dim: int = None, act_func: str = 'relu'):
        """This function adds a hidden layer to the network

            layer size: the number of neurons for the current layer

            input_dim: the dimension of the first input layer, it has to be passed as a parameter if there are no
            layers created yet

            act_func: the activation function for the current layer
        """
        if self._input_dim == -1 and input_dim is None:
            raise Exception("The first layer's input must be specified!")

        self.activation_func.append(ACTIVATION_FUNCTIONS[act_func])
        self.deriv_activation_func.append(DERIV_ACTIVATION_FUNCTION[act_func])

        # connect the two layers and create a matrix of randomized weights
        if self._input_dim == -1:
            self._input_dim = input_dim
            self.layers.append(np.random.uniform(low=-1, high=1, size=(
                layer_size, input_dim)))
        else:
            self.layers.append(np.random.uniform(low=-1, high=1, size=(layer_size, self.layers[-1].shape[0])))
        # create the bias vector for the current layer
        self.bias.append(np.zeros((layer_size, 1)))

    def _forward_prop(self, X):
        """This function implements the forward propagation through the layers of a neural network

            X: the input from the training dataset of shape (n, m)

            :returns a list 'Z' of the outputs of each layer before going through the activation function
                    a list 'A' of the outputs of each layer after the activation function
        """
        Z = []
        A = []
        a = X.transpose()
        # traverse through the layers from left to right and calculate the output of the network
        for W, b, phi in zip(self.layers, self.bias, self.activation_func):
            z = np.matmul(W,
                          a) + b  # multiplying the output vector of the previous layer with the weight matrix connecting the two layers
            Z.append(z)
            a = phi(z)  # pass the result of the through the activation function
            A.append(a)
        return Z, A

    def _back_prop(self, A, Z, X, Y):
        """This function implements the backwards propagation through the layers of a neural network

            A: a list of vectors that hold the outputs of each layer after the activation function

            Z: a list of vectors that hold the outputs of each layer before going through the activation function

            X: the input from the training dataset

            Y: the labels for the training dataset

            :returns the deltas for the weights and biases of each layers
        """

        dW = []
        dB = []

        # Handle the outer layer first. A[-1] contains the predicted outputs and Y are the true ones.
        # the subtraction is based on the combined derivative of the cross-entropy and softmax function
        dw_next = A[-1] - Y.transpose()
        m = Y.shape[0]
        # traverse from right to left and calculate the deltas for the weights and for the biases
        # the error form the output layer is passed through the network and the delta calculations
        # are held in the dW and dB lists
        for i in range(len(self.layers) - 1, 0, -1):
            W = self.layers[i]

            dB.insert(0, 1 / m * np.sum(dw_next, axis=1))
            dW.insert(0, 1 / m * np.matmul(dw_next, A[i - 1].transpose()))
            dw_next = (np.matmul(dw_next.transpose(), W) * self.deriv_activation_func[i](
                Z[i - 1]).transpose()).transpose()
        dW.insert(0, 1 / m * np.matmul(dw_next, X))
        dB.insert(0, 1 / m * np.sum(dw_next, axis=1))

        return dW, dB

    def _update_weights(self, dW, dB, lr):
        """This functions updates the weights and biases of each layers using gradient descent

            dW: the list of deltas to be subtracted form the weight matrix of each layer

            dB: the list of deltas to be subtracted from the biases vector of each layer
        """

        # traverse the network from left to right and update the weights of each layer
        for i in range(len(self.layers)):
            self.layers[i] = self.layers[i] - lr * dW[i]
            self.bias[i] = self.bias[i] - lr * dB[i].reshape((dB[i].shape[0], 1))

    def _grad_descent(self, X, Y, batch_size, epochs, lr):
        """This function implements the gradient descent algorithm for optimizing the loss function of the last layer

            X: the input from the training dataset

            Y: the labels for the training dataset

            batch_size: the number of sample to be passed in each propagation

            epochs: the number of epochs during training

            lr: the learning rate
        """
        X_axis = []
        Y_test_axis = []
        Y_train_axis = []
        total_time = 0
        for iteration in range(epochs):
            start = time.time()
            # Train the model in batches, for each epoch. Each batch goes through
            # forward, backward propagation and update weights
            for i in range(0, X.shape[0] - batch_size, batch_size):
                x = X[i:i + batch_size]
                y = Y[i:i + batch_size]
                Z, A = self._forward_prop(x)
                dW, dB = self._back_prop(A, Z, x, y)
                self._update_weights(dW, dB, lr)
            end = time.time()
            total_time += end - start
            if self._verbose or self._plot_learning_curve:
                acc_score = accuracy_score(self.one_hot.inverse_transform(Y), self.predict(X))
                if self._verbose:
                    print('Epoch #', iteration + 1, '/', epochs, ', accuracy: ',
                          '{:.2f}'.format(acc_score * 100), '%',
                          ' time elapsed: ',
                          '{:.2f}'.format((end - start)), 's')
                if self._plot_learning_curve:
                    X_axis.append(iteration)
                    Y_train_axis.append(acc_score)
                    Y_test_axis.append(accuracy_score(y_test, self.predict(X_test)))
        # Plotting the learning curve
        if self._plot_learning_curve:
            plt.title('Learning curve')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.grid()
            plt.plot(X_axis, Y_train_axis, 'o-', color='red', label='Training curve')
            plt.plot(X_axis, Y_test_axis, 'o-', color='green', label='Testing curve')
            plt.legend(loc='best')
            plt.show()
        print('Total fitting time: ', '{:.4f}'.format(total_time), 's')

    def fit(self, X, Y, batch_size=100, epochs=10, lr=0.1):
        """This function initiates the training process and fits the data to the model

            X: the input from the training dataset

            Y: the labels for the training dataset

            batch_size: the number of sample to be passed in each propagation

            epochs: the number of epochs during training

            lr: the learning rate
        """
        if self._input_dim == -1:
            raise Exception('There are no layers created')
        self.one_hot.fit(Y)  # one hot encode the labels vector to
        self.add_layer(len(self.one_hot.classes_),
                       act_func='softmax')  # adding the output layer before initiating the training process
        self._grad_descent(X.to_numpy(), self.one_hot.transform(Y), batch_size, epochs, lr)

    def predict(self, X):
        """This function given an instance X predicts the label Y

            X: the input from the training dataset

            :returns the predicted label/s for input X
        """
        X = np.array(X)
        _, output = self._forward_prop(X)
        predictions = np.argmax(output[-1].transpose(), axis=1)  # choosing the class with the highest probability
        return predictions


if __name__ == '__main__':
    # # read the data from the csv file
    # df = pd.read_csv('dataset.csv')
    # print('DATASET READ COMPLETE...')
    #
    # # shuffle the dataset
    # df = df.sample(frac=1)
    #
    # # separate the data from the label and normalize the values between 0 and 1
    # X_data = df.drop(['label'], axis=1) / 255.0
    # y_data = df['label']
    #
    # # split the data to a train (60%) and a test set (40%)
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
    # print('DATASET SPLIT...')
    # print()
    data = loadmat('mnist_all.mat')
    X_train = pd.DataFrame()
    y_train = []
    X_test = pd.DataFrame()
    y_test = []
    for i in range(10):
        train_digit = pd.DataFrame(data['train' + str(i)])
        test_digit = pd.DataFrame(data['test' + str(i)])

        train_labels = np.full(train_digit.shape[0], i)
        test_labels = np.full(test_digit.shape[0], i)
        X_train = X_train.append(train_digit, True)
        X_test = X_test.append(test_digit, True)

        y_train.extend(train_labels)
        y_test.extend(test_labels)

    y_train = pd.DataFrame(y_train, columns=['label'])
    y_test = pd.DataFrame(y_test, columns=['label'])

    train_samples = X_train.join(y_train).sample(frac=1)
    test_samples = X_test.join(y_test).sample(frac=1)

    X_train = train_samples.drop(['label'], axis=1) / 255.0
    y_train = train_samples['label']
    X_test = test_samples.drop(['label'], axis=1) / 255.0
    y_test = test_samples['label']

    # Train with 1 hidden layer of 64 neurons
    print('MLP, 1 hidden layer 64 neurons, batch size 80, epochs 10')
    classifier = MLPmodel(verbose=True, plc=True)
    classifier.add_layer(10, 784)
    classifier.fit(X_train, y_train, batch_size=20, epochs=10)
    print('Test accuracy: ', '{:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test)) * 100), '%')
    print()

    # Train with 1 hidden layer of 280 neurons
    print('MLP, 1 hidden layer 280 neurons, batch size 80, epochs 10')
    classifier = MLPmodel(verbose=True, plc=True)
    classifier.add_layer(280, 784)
    classifier.fit(X_train, y_train, batch_size=80, epochs=10)
    print('Test accuracy: ', '{:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test)) * 100), '%')
    print()

    # Train with 1 hidden layer of 64 neurons and one of 16
    print('MLP, 1 hidden layer 64 neurons, 1 hidden layer 16 neurons, batch size 30, epochs 20')
    classifier = MLPmodel(verbose=True, plc=True)
    classifier.add_layer(64, 784)
    classifier.add_layer(16)
    classifier.fit(X_train, y_train, batch_size=30, epochs=20)
    print('Test accuracy: ', '{:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test)) * 100), '%')
    print()

    # Train with 1 hidden layer of 42 neurons
    print('MLP, 1 hidden layer 42 neurons, batch size 50, epochs 50')
    classifier = MLPmodel(verbose=True, plc=True)
    classifier.add_layer(42, 784)
    classifier.fit(X_train, y_train, batch_size=50, epochs=50)
    print('Test accuracy: ', '{:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test)) * 100), '%')
    print()

    # Nearest centroid classifier
    print('Nearest centroid classifier')
    start = time.time()
    classifier = NearestCentroid()
    classifier.fit(X_train, y_train)
    end = time.time()
    print('Train accuracy: ', '{:.2f}'.format(accuracy_score(y_train, classifier.predict(X_train)) * 100), '%')
    print('Test accuracy: ', '{:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test)) * 100), '%')
    print('Fitting time: ', end - start, 's')
    print()

    # Nearest Neighbour classifier n_neighbours = 5
    print('K nearest neighbor classifier for 5 neighbors')
    classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    start = time.time()
    classifier.fit(X_train, y_train)
    end = time.time()
    # print('Train accuracy: ', '{:.2f}'.format(accuracy_score(y_train, classifier.predict(X_train)) * 100), '%')
    print('Test accuracy: ', '{:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test)) * 100), '%')
    print('Fitting time: ', end - start, 's')
    print()


