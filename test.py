from neuralnetwork import NeuralNetwork
from dataloader import load_data
from functions import ReLU, Softplus, Softmax, CE_Loss

if __name__ == "__main__":
    train_images, train_labels = load_data("mnist_train.csv")
    test_images, test_labels = load_data("mnist_test.csv")

    print("Data loaded.")
    
    # Example setup
    layers = [784, 256, 10]
    activations = [Softplus, Softmax]
    learning_rate = 1e-3
    
    network = NeuralNetwork(len(layers), layers, activations, learning_rate)

    network.train(train_images, train_labels, CE_Loss)

    network.test(test_images, test_labels, CE_Loss)