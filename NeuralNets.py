"""
Neural Nets Tools
Each Layer objects have the reference to previous layer and next layer
Each layer manages it's own n_neurons, including passing output of previous layer to all child n_neurons and collect the
output of these n_neurons and pass them to next layer
"""
from abc import abstractmethod

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
VERBOSE_ARGS = {'PRINT_SKIP_ITER': 100}


def sigmoid(x, dx=False):
    if dx:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def linear(x, dx=False):
    if dx:
        return 1.0
    return x


class Neuron:
    def __init__(self, n_dim, activation_func=sigmoid, layer=None):
        """

        :param n_dim: dimension of input
        :param activation_func: activation function, python function object
        :param layer: Layer object that holds this neuron
        """
        self.pre_synapse_excites = 0
        self.layer = layer
        self.weight = np.random.uniform(low=-0.3, high=0.3, size=n_dim + 1)
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
            self.delta = err * self.activation_func(self.pre_synapse_excites, dx=True)
            return self.delta
        else:
            self.delta = self.activation_func(self.pre_synapse_excites, dx=True) * delta_next
            return self.delta

    def update(self, lr=0.5, momentum=0.9):
        """
        update function of neurons, update the weight based on pre-calculated delta value
        - should be invoked after back_propagate is invoked
        - should be invoked by the governing Layer object
        :param lr: learning rate of backprop
        :param momentum: momentum term
        :return:
        """
        update = lr * self.delta * self.augmented_input
        new_weight = self.weight + update + momentum * self.last_update
        self.last_update = update
        self.weight = new_weight

    def print_weight(self):
        print(self.weight)


class Layer:
    """
    General layer object
    each layer will keep track of the previous layer and next layer so that recursive function call could be implemented
    """

    def __init__(self, n_neurons, name=None):
        self.n_neurons = n_neurons
        self.next_layer = None
        self.name = name

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    @abstractmethod
    def forward_propagate(self, input):
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Dense layer (Fully-connected Feed-forward layer)
    """

    def __init__(self, n_neurons, previous_layer, isOutput=False, activation_function=sigmoid, name=None):
        """
        Initialize a DenseLayer object
        :param n_neurons: number of neurons of current layer
        :param previous_layer: previous layer, whose n_neurons will be used as input dimension for this layer
        :param isOutput: bool value to indicate if this layer is a output layer
        :param activation_function: activation function, function object
        """
        Layer.__init__(self, n_neurons, name)
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
        """
        forward_propagate method
        For DenseLayer object, this method should only be invoked by another Layer object as an action of recursive call
        Calculate the output by first applying the dot-product of previous_payer_output with each neuron's
        weight vector and apply activation function. The output of each neuron is concatenated and passed to next layer
        :param previous_layer_output: np.array with dimensionality = (self.n_neurons)
        :return:
        """
        output = []
        for i in range(len(self.neurons)):
            output.append(self.neurons[i].forward_propagate(previous_layer_output))
        output = np.array(output)
        if self.isOutput:
            return output
        else:
            return self.next_layer.forward_propagate(output)

    def back_propagate(self, error=None):
        """
        Back propagate algorithm, update delta for each neuron
        :param error: difference of actual output of expected output, should be None if this is not the output layer
        :return:
        """
        delta_ = []
        if self.isOutput and error is not None:
            # if the current layer is not the output layer, then we can calculate delta of the neurons based on diff of
            # actual output and expected output
            for i in range(len(self.neurons)):
                delta_neuron = self.neurons[i].back_propagate(err=error[i], isOutput=True)
                delta_.append(delta_neuron)
            self.delta_ = delta_
            self.previous_layer.back_propagate()

        elif not self.isOutput:
            # if the current layer is somehow a hidden layer, then the delta is calculated based on weighted sum of delta
            # from next layer
            for i in range(len(self.neurons)):
                delta_neuron = self.neurons[i].back_propagate(delta_next=self.get_weighted_delta_sum(i))
                delta_.append(delta_neuron)
            self.delta_ = delta_
            self.previous_layer.back_propagate()

    def update(self, lr=0.5, momentum=0.9):
        """
        After calculating the delta, update function should be called to actually update the weight of each neuron
        The update method should be invoked at the output layer and will be executed to the entire network at a recursive
        order
        :param lr: learning rate
        :param momentum: momentum term, set to 0 to disable momentum
        :return:
        """
        for neuron in self.neurons:
            neuron.update(lr=lr, momentum=momentum)
        self.previous_layer.update(lr=lr, momentum=momentum)

    def get_weighted_delta_sum(self, index):
        """
        get weighted sum of next layer delta based on index
        the weighted sum is calculated as the weighted sum of the synaptic weight of index th neuron of current layer to
        each neuron of next layer and their delta
        :param index: int; index of a neuron inside this layer
        :return: float; weighted sum of delta
        """
        if self.isOutput:
            return
        sum = 0
        for i in range(len(self.next_layer.neurons)):
            sum += np.multiply(self.next_layer.neurons[i].weight[index], self.next_layer.neurons[i].delta)
        return sum

    def print_weight(self):
        print('--------------------------------')
        if self.name is not None:
            print("Layer " + self.name)
        print("Layer type: Dense")
        print("n_neurons: " + str(self.n_neurons))
        for i in range(len(self.neurons)):
            self.neurons[i].print_weight()

        if self.isOutput:
            print('--------------------------------')
            return
        self.next_layer.print_weight()


class InputLayer(Layer):
    """
    InputLayer object
    """

    def __init__(self, n_neurons, name=None):
        """

        :param n_neurons: dimensionality of input
        """
        Layer.__init__(self, n_neurons, name=name)
        self.next_layer = None

    def forward_propagate(self, input):
        """
        forward_propagate
        To do a prediction using the current model, should invoke forward_propagate at the InputLayer
        Then the calculation is completed recursively
        :param input: np.array; input data, should have the same dimensionality with self.n_neurons
        :return: np.array; output of the output layer, should have same dimensionality with output layer's n_neurons
        """
        return self.next_layer.forward_propagate(input)

    def back_propagate(self):
        """
        back_propagate
        Back propagate function for each layer. For InputLayer is just a sign of termination of back_propagate.
        This function should only be called as a termination step of recursive execution of back_propagate by the next_layer
        of this InputLayer
        :return:
        """
        return

    def update(self, lr=0.5, momentum=0.9):
        """
        update
        Weight update function for each layer. For InputLayer is just a sign of termination of update call stack.
        This function should only be called as a termination step of recursive execution of update by the next_layer
        of this InputLayer
        :param lr:
        :param momentum:
        :return:
        """
        return

    def print_weight(self):
        print('--------------------------------')
        if self.name is not None:
            print("Layer " + self.name)
        print("Layer type: Input")
        print("dim: " + str(self.n_neurons))
        self.next_layer.print_weight()


class Model:
    """
    Model object that wraps the layers and provide model level functions
    """

    def __init__(self, input_layer, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer

    def err_(self, actual_output, desired_output):
        """
        Calculate difference of actual output and desired output
        which is desired_output - actual output
        :param actual_output: np.array; actual_output
        :param desired_output: np.array; desired output
        :return: np.array; with same dimensionality as desired_output
        """
        return np.subtract(desired_output, actual_output)

    def abs_err_(self, actual_ouput, desired_output):
        """
        Calculate difference of actual output and desired output
        which is abs(desired_output - actual output)
        :param actual_ouput: np.array; actual output of the neural net
        :param desired_output: np.array; desired output
        :return: np.array; with same dimensionality as desired_output
        """
        return np.absolute(np.subtract(desired_output, actual_ouput))

    def fit(self, Xs, Ys, lr=0.5, momentum=0.9, max_epochs=10000, tolerance=0.05, verbose=0):
        """
        Fit model on a set of data, this is not batch optimization
        :param Xs: np.array, set of attributes
        :param Ys: np.array, set of desired output
        :param lr: learning rate
        :param momentum: momentum
        :param max_epochs: maximum epochs that limits the training iteration
        :param tolerance: tolerance of early stopping
        :return: list; log of average absolute difference of desired of output and actual output
        """
        if Xs.shape[0] != Ys.shape[0]:
            return
        log = []
        for epoch in range(max_epochs):
            for i in range(Xs.shape[0]):
                # loop for updating weights
                x = Xs[i, :]
                y = Ys[i, :]
                y_ = self.input_layer.forward_propagate(input=x)
                err = self.err_(actual_output=y_, desired_output=y)
                self.output_layer.back_propagate(error=err)
                self.output_layer.update(lr=lr, momentum=momentum)
            early_stop = True
            loss_sum = np.zeros(Ys[0, :].shape)
            for i in range(Xs.shape[0]):
                # loop for early stop
                x = Xs[i, :]
                y = Ys[i, :]
                y_ = self.input_layer.forward_propagate(input=x)
                abs_error = self.abs_err_(actual_ouput=y_, desired_output=y)
                loss_sum += abs_error
                if np.greater(abs_error, tolerance).any():
                    early_stop = False
            loss_sum /= Ys.shape[0]
            log.append(loss_sum)
            if verbose == 1:
                if epoch % VERBOSE_ARGS['PRINT_SKIP_ITER'] == 0:
                    print(str(epoch) + "th iteration")
                    self.print_weight()
                    print(self.predict(Xs))
                    print(Ys)
            if early_stop:
                break

        return log

    def predict(self, Xs):
        result = []
        for i in range(Xs.shape[0]):
            result.append(self.input_layer.forward_propagate(Xs[i, :]))
        return np.array(result)

    def print_weight(self):
        print('=============================================================')
        self.input_layer.print_weight()


class KmeansClustering:
    def __init__(self, k_centers=5):
        self.k_centers = k_centers
        self.centers = []
        self.size_data = 0

    def fit(self, data, max_iter=10000, epsilon=0.01):
        self.size_data = data.shape[0]
        if self.k_centers > self.size_data:
            return
        else:
            # initialize k centers randomly
            for i in np.random.choice(range(self.size_data), self.k_centers):
                self.centers.append(data[i, :])
        # iterations for update the clusters' centers
        for i in range(0, max_iter):
            # get datapoints into the clusters
            clusters = self.cluster(data)
            # get new centers
            new_centers = []
            for ii in range(len(clusters)):
                new_centers.append(np.average(clusters[ii][0], axis=0))
            if self.early_stop(previous_centers=self.centers, new_centers=new_centers,epsilon=epsilon):
                return clusters
            self.centers = new_centers

    def cluster(self, data):
        """
        cluster the data points into clusters
        :param data: set of data points, np.array
        :return: list of clusters, each cluster is a tuple, consisting of a list of indexes of datapoints and the variance
        if a cluster contains only 1 datapoint, it'll be set to the mean of all other variances
        """
        size_data = data.shape[0]
        clusters = self.k_centers * [None]
        for i in range(len(clusters)):
            clusters[i] = []
        cluster_distance_sum = self.k_centers * [0.0]
        # assign each datapoint to one of the clusters
        for i in range(size_data):
            datapoint = data[i, :]
            cluster, dist = self.assign_cluster(datapoint)
            clusters[cluster].append(datapoint)
            cluster_distance_sum[cluster] += dist
        # calculate the variance of the clusters
        sum_var = 0.0
        total_non_trival_cluster = 0
        for i in range(self.k_centers):
            if len(clusters[i]) <= 1:
                variance = None
            else:
                variance = cluster_distance_sum[i] / len(clusters[i])
                sum_var += variance
                total_non_trival_cluster += 1
            clusters[i] = (clusters[i], variance)
        ave_var = sum_var / float(total_non_trival_cluster)
        for i in range(self.k_centers):
            if clusters[i][1] is None:
                clusters[i] = (clusters[i][0], ave_var)
                # clusters[i][1] = ave_var
        return clusters

    def assign_cluster(self, data_point):
        """
        assign a single datapoint into the nearest cluster
        :param data_point:
        :return: tuple(index, float) which is the index of the cluster and the squared euclidean distance
        """
        target_index = 0
        minimum_distance_sqr = 99999.9
        for i in range(self.k_centers):
            distance_sqr = self.squared_euclidean_distance(self.centers[i], data_point)
            if distance_sqr < minimum_distance_sqr:
                minimum_distance_sqr = distance_sqr
                target_index = i
        return target_index, minimum_distance_sqr

    @staticmethod
    def squared_euclidean_distance(a, b):
        """
        calculate squared euclidean distance between datapoint a and b
        :param a: np.array
        :param b: np.array
        :return: scalar
        """
        return np.square(np.linalg.norm(a - b))

    @staticmethod
    def early_stop(previous_centers, new_centers, epsilon):
        """
        calculate the squared Euclidean distance for each pair of old & new cluster centers, if all of the pairs has
        Euclidean distance smaller than epsilon, it'll return True to trigger early stop, otherwise will return False
        to push the algorithm not stopping
        :param previous_centers: list of nparrays, list of centers of different clusters during previous iteration
        :param new_centers: list of nparrays, which is a list of centers of different clusters
        :param epsilon: float, the tolerance of difference
        :return: boolean, True for early stop, False for keep iteration
        """
        for i in range(len(previous_centers)):
            if KmeansClustering.squared_euclidean_distance(previous_centers[i], new_centers[i]) <= epsilon:
                return True
        return False


# sample test code
if False:
    input = InputLayer(3)
    hid1 = DenseLayer(5, input)
    hid2 = DenseLayer(2, hid1, isOutput=True)

    mdl = Model(input, hid2)

    input_value = np.random.rand(2, 3)
    desired_output = np.array([[1, 0], [1, 1]])
    log = mdl.fit(Xs=input_value, Ys=desired_output, max_epochs=10000, verbose=1)
    print(mdl.predict(input_value))
    print(log)
if False:
    import dataGenerator

    dg = dataGenerator.ParityDataGenerator()
    df = dg.generate_data()


    def get_base_model_obj():
        input = InputLayer(n_neurons=4)
        hid1 = DenseLayer(n_neurons=4, previous_layer=input, activation_function=sigmoid)
        # hid1 = NN.DenseLayer(n_neurons=4, previous_layer=hid1)
        hid2 = DenseLayer(n_neurons=1, previous_layer=hid1, isOutput=True)
        mdl = Model(input_layer=input, output_layer=hid2)
        return mdl


    VERBOSE_ARGS['PRINT_SKIP_ITER'] = 1000
    logs = []
    lens = []
    df = df.sample(frac=1.0)
    for lr in np.arange(0.05, 0.04, -0.05):
        mdl = get_base_model_obj()
        logs = mdl.fit(df.drop(columns=['label']).values, df['label'].values.reshape(-1, 1), lr=lr, momentum=0,
                       max_epochs=10000000, verbose=1, tolerance=0.05)
        print(logs[-1])
        lens.append((lr, len(logs)))

    with open('./log.txt', 'w') as file:
        file.write(str(lens))
    print(lens)
