from neuralnetwork import Network

if __name__ == "__main__":
    
    # Example setup
    layers = [784, 256, 64, 10]
    activations = ["relu", "relu", "softmax"]
    
    network = Network(layers, activations)