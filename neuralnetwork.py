import numpy as np

class NeuralNetwork:
    def __init__(self, num_layers, neurons, activations):
        # Make sure the parameters match properly
        assert num_layers == len(neurons) == (len(activations) + 1)

        # num_layers is an int representing how many fully connected layers there are in the network
        self.num_layers = num_layers

        # neurons is a list like [784, 256, 10] which represents the number of neurons per layer
        self.neurons = neurons

        # activations is a list of functions to use for each layer (not including the input layer)
        self.activations = activations

        # self.layers is a list of 2D numpy arrays, each array has dimension nxd,
        # where n = number of neurons in the layer and d = number of neurons in the next layer
        # Meaning each neuron in a layer has d weights
        self.layers = []

        # self.biases is a list of 1D numpy arrays, each array has dimension n, 
        # where n = number of neurons in the layer, meaning each neuron in a layer has 1 bias
        self.biases = []

        # Randomly initialize neuron weights and bias for each layer
        for i in range(self.num_layers - 1):
            neurons = np.random.normal(0, 1, (self.neurons[i], self.neurons[i + 1]))

            # No bias for the raw inputs
            bias = np.random.normal(0, 1, (self.neurons[i + 1], ))

            self.layers.append(neurons)
            self.biases.append(bias)        
    
    # Dot product on inputs & weights + bias -> activation function -> output
    def fire_neurons(self, inputs, weights, bias, activation):
        return activation((inputs @ weights) + bias)

    # Make a prediction based on the given inputs
    def forward_pass(self, inputs):
        # Flatten (28, 28) images into a (784, ) array
        values = inputs.reshape(784, )

        # Pass values from each layer up
        for i in range(len(self.layers)):
            values = self.fire_neurons(values, self.layers[i], self.biases[i], self.activations[i])

        return values