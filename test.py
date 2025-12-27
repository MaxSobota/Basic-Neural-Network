from neuralnetwork import NeuralNetwork
from dataloader import load_data
from activations import relu, softmax
import numpy as np

if __name__ == "__main__":
    images, labels = load_data()
    
    # Example setup
    layers = [784, 256, 64, 10]
    activations = [relu, relu, softmax]
    
    network = NeuralNetwork(len(layers), layers, activations)

    probabilities = network.forward_pass(images[5])

    most_likely = np.argmax(probabilities)

    print(f"Prediction probabilities: {probabilities} Actual value: {labels[5]} Prediction: {most_likely}")