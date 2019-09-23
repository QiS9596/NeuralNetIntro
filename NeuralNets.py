"""
Neural Nets Tools
Each Layer objects have the reference to previous layer and next layer
Each layer manages it's own n_neurons, including passing output of previous layer to all child n_neurons and collect the
output of these n_neurons and pass them to next layer
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from abc import abstractmethod
matplotlib.use('TkAgg')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, n_dim, activation_func=sigmoid):
        """

        :param n_dim: dimension of input
        :param activation_func: activation function, python function object
        """
        self.weight = np.random.rand(n_dim + 1)
        self.activation_func = activation_func

    def forward_propagate(self, input):
        """
        Forward propagate of a single neuron
        input should be the input or the output of previous layer
        :param input: np.array with n_dim*1, should be passed via governering Layer object
        :return: the output of current neuron
        """
        augmented_input = np.concatenate([input, 1], axis=None)
        return self.activation_func(np.dot(augmented_input, self.weight.T))


class Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.next_layer = None

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    @abstractmethod
    def forward_paropagate(self, input):
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, n_neurons, previous_layer, isOutput=False):
        Layer.__init__(self, n_neurons)
        self.isOutput = isOutput
        self.next_layer = None
        self.previous_layer = previous_layer
        self.previous_layer.set_next_layer(self)
        self.neurons = []
        for i in range(self.n_neurons):
            self.neurons.append(Neuron(n_dim=self.previous_layer.n_neurons))

    def forward_propagate(self, previous_layer_output):
        output = []
        for neuron in self.neurons:
            output.append(neuron.forward_propagate(previous_layer_output))
        output = np.array(output)
        if self.isOutput:
            return output
        else:
            return self.next_layer.forward_propagate(output)


class InputLayer(Layer):
    def __init__(self, n_neurons):
        Layer.__init__(self, n_neurons)
        self.next_layer = None

    def forward_propagate(self, input):
        return self.next_layer.forward_propagate(input)

