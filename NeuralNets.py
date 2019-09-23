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


def sigmoid(x, dx=False):
    if dx:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, n_dim, activation_func=sigmoid, layer=None):
        """

        :param n_dim: dimension of input
        :param activation_func: activation function, python function object
        :param layer: Layer object that holds this neuron
        """
        self.pre_synapse_excites = 0
        self.layer = layer
        self.weight = np.random.rand(n_dim + 1)
        self.activation_func = activation_func
        self.last_update = 0
        self.delta = 0
        self.augmented_input = None

    def forward_propagate(self, input):
        """
        Forward propagate of a single neuron
        input should be the input or the output of previous layer
        :param input: np.array with n_dim*1, should be passed via governering Layer object
        :return: the output of current neuron
        """
        augmented_input = np.concatenate([input, 1], axis=None)
        self.pre_synapse_excites = np.dot(augmented_input, self.weight.T)
        self.augmented_input = augmented_input
        return self.activation_func(self.pre_synapse_excites)

    def back_propagate(self, delta_next=None, isOutput=False, err=None):
        """

        :param delta_next: weighted sum of delta, calculated at each layer object
        :param isOutput: if the neuron belongs to output layer set to true, then the calculation of delta is also different
        :param err: error value for calculating the last layer
        :return: delta value for this neuron
        """
        if isOutput and err is not None:
            self.delta = err*self.activation_func(self.pre_synapse_excites, dx=True)
            return self.delta
        else:
            self.delta = self.activation_func(self.pre_synapse_excites, dx=True)*delta_next
            return self.delta

    def update(self, lr=0.5, momentum=0.9):
        pass


class Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.next_layer = None

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    @abstractmethod
    def forward_propagate(self, input):
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(self, n_neurons, previous_layer, isOutput=False, activation_function=sigmoid):
        Layer.__init__(self, n_neurons)
        self.isOutput = isOutput
        self.next_layer = None
        self.previous_layer = previous_layer
        self.previous_layer.set_next_layer(self)
        self.neurons = []
        self.delta_ = []
        for i in range(self.n_neurons):
            self.neurons.append(
                Neuron(n_dim=self.previous_layer.n_neurons, activation_func=activation_function, layer=self))

    def forward_propagate(self, previous_layer_output):
        output = []
        for i in range(len(self.neurons)):
            output.append(self.neurons[i].forward_propagate(previous_layer_output))
        output = np.array(output)
        if self.isOutput:
            return output
        else:
            return self.next_layer.forward_propagate(output)

    def back_propagate(self, error=None):
        delta_ = []
        if self.isOutput and error is not None:
            for i in range(len(self.neurons)):
                delta_neuron = self.neurons[i].back_propagate(err=error[i], isOutput=True)
                delta_.append(delta_neuron)
            self.delta_ = delta_
            self.previous_layer.back_propagate()

        elif not self.isOutput :
            for i in range(len(self.neurons)):
                delta_neuron = self.neurons[i].back_propagate(delta_next=self.get_weighted_delta_sum(i))
                delta_.append(delta_neuron)
            self.delta_ = delta_
            self.previous_layer.back_propagate()


    def get_weighted_delta_sum(self, index):
        if self.isOutput:
            return
        sum = 0
        for i in range(len(self.next_layer.neurons)):
            sum += np.multiply(self.next_layer.neurons[i].weight[index], self.next_layer.neurons[i].delta)
        return sum


class InputLayer(Layer):
    def __init__(self, n_neurons):
        Layer.__init__(self, n_neurons)
        self.next_layer = None

    def forward_propagate(self, input):
        return self.next_layer.forward_propagate(input)

    def back_propagate(self):
        return

input = InputLayer(3)
hid1 = DenseLayer(5,input)
hid2 = DenseLayer(3, hid1,isOutput=True)
input.forward_propagate(np.random.rand(3))
hid2.back_propagate(np.random.rand(3))