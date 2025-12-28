import numpy as np

class NeuralNetwork:
    def __init__(self, num_layers, layers, activations, learning_rate):
        # Make sure the parameters match properly
        assert num_layers == len(layers) == (len(activations) + 1)

        # num_layers is an int representing how many fully connected layers there are in the network
        self.num_layers = num_layers

        # neurons is a list like [784, 256, 10] which represents the number of neurons per layer
        self.neurons = layers

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
            neurons = np.random.normal(0, 0.5, (self.neurons[i], self.neurons[i + 1]))

            # No bias for the raw inputs
            bias = np.random.normal(0, 0.5, (self.neurons[i + 1], ))

            self.layers.append(neurons)
            self.biases.append(bias)        

        # Controls how big gradient descent steps are
        self.learning_rate = learning_rate
    
    # Make a prediction based on the given inputs
    def forward_pass(self, inputs):
        # Flatten (28, 28) images into a (784, ) array
        values = inputs.flatten()

        # Save these for backpropagation
        self.layer_inputs = []
        self.pre_activations = []
        self.post_activations = []

        # Pass values from each layer up
        for i in range(len(self.layers)):
            # Save input activation
            self.layer_inputs.append(values)

            # Dot product + bias
            z = values @ self.layers[i] + self.biases[i]
            self.pre_activations.append(z)

            # Use activation function
            values = self.activations[i].func(z)
            self.post_activations.append(values)

        return values
    
    def backward_pass(self, labels, loss_derivative):
        weight_gradients = []
        bias_gradients = []

        # After softmax = prediction
        delta = loss_derivative(self.post_activations[-1], labels)

        # Work back to front
        for i in reversed(range(len(self.layers))):
            # Previous layer derivative
            a_previous = self.layer_inputs[i]

            # Weight/bias gradients
            weight_gradients.append(np.outer(a_previous, delta))
            bias_gradients.append(delta)

            # Stop if we reached the first layer
            if i == 0:
                break

            # Propagate error signal backward:
            error_signal = delta @ self.layers[i].T

            # Previous layer activation derivative
            da_dz = self.activations[i - 1].func_derivative(self.pre_activations[i - 1])

            # New delta
            delta = error_signal * da_dz

        # Reverse because we built them in reverse
        weight_gradients.reverse()
        bias_gradients.reverse()

        return weight_gradients, bias_gradients
    
    # Push the weights/biases towards lower loss, step size is based on learning rate
    def update_parameters(self, weight_gradients, bias_gradients):
        for i in range(len(self.layers)):
            self.layers[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    # Learn by doing forward/backward pass on dataset
    def train(self, inputs, labels, loss):
        print("Beginning training...")

        losses = np.zeros((len(labels), ))
        correct = 0

        for i in range(len(labels)):
            prediction = self.forward_pass(inputs[i])
            losses[i] = loss.func(prediction, labels[i])

            if np.argmax(prediction) == np.argmax(labels[i]):
                correct += 1

            weight_gradients, bias_gradients = self.backward_pass(labels[i], loss.func_derivative)

            # Update parameters to improve model
            self.update_parameters(weight_gradients, bias_gradients)

        print(f"Average loss across {len(labels)} images: {np.mean(losses)}")
        print(f"Total correct: {correct} / {len(labels)}")

    # Compare predictions to true labels without learning
    def test(self, inputs, labels, loss):
        print("Beginning testing...")

        losses = np.zeros((len(labels), ))
        correct = 0

        for i in range(len(labels)):
            prediction = self.forward_pass(inputs[i])
            losses[i] = loss.func(prediction, labels[i])

            if np.argmax(prediction) == np.argmax(labels[i]):
                correct += 1

        print(f"Average loss across {len(labels)} images: {np.mean(losses)}")
        print(f"Total correct: {correct} / {len(labels)}")