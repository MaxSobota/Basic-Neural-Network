from neuralnetwork import NeuralNetwork
from dataloader import load_data, Dataset
from functions import ReLU, Softplus, Softmax, CE_Loss
from visualizer import start

def example_setup():
    # Example setup
    layers = [784, 256, 10]
    activations = [Softplus, Softmax]
    learning_rate = 1e-3
    
    network = NeuralNetwork(len(layers), layers, activations, learning_rate)

    return network

if __name__ == "__main__":
    train_images, train_labels = load_data("mnist_train.csv")
    train_dataset = Dataset(train_images, train_labels)
    
    test_images, test_labels = load_data("mnist_test.csv")
    test_dataset = Dataset(test_images, test_labels)

    start(train_dataset, test_dataset)

# TODO: Add progress bar to training
# TODO: Generalize network for any problem
# TODO: Try pooling to make a smaller network?
# TODO: Fix network activation colors
# TODO: Clean up and finish GUI
# TODO: Figure out how to make visualizer look better
# TODO: Add training/testing/saving models
# TODO: Add drawing features