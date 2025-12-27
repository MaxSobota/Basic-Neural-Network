import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation):
        pass

class Network:
    def __init__(self, layers, activations):
        self.layers = []

        for layer in layers:
            for i in range(layer):
                self.layers.append(Neuron())