"""
Neural Nets Tools
Each Layer objects have the reference to previous layer and next layer
Each layer manages it's own neurons, including passing output of previous layer to all child neurons and collect the
output of these neurons and pass them to next layer
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, n_dim, activation_func=sigmoid):
        self.weight = np.random.rand(n_dim+1)
        self.activation_func = activation_func
    def forward_propagate(self, input):
        """
        Forward propagate of a single neuron
        input should be the input or the output of previous layer
        :param input: np.array with n_dim*1, should be passed via governering Layer object
        :return: the output of current neuron
        """
        augmented_input = np.concatenate([input, 1])
        return self.activation_func(np.dot(augmented_input,self.weight))


class Layer:
    def __init__(self, neurons):
        pass

class DenseLayer(Layer):
    def __init__(self, neurons, previous_layer):
        Layer.__init__(self, neurons)
        self.next_layer = None
        self.weight_vector = None
        pass

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def forward_propagate(self, previous_layer_output):
        pass


