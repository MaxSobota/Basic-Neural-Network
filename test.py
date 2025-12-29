from neuralnetwork import NeuralNetwork
from dataloader import load_data, save_network, load_network
from functions import ReLU, Softplus, Softmax, CE_Loss

def example_setup():
    # Example setup
    layers = [784, 256, 10]
    activations = [Softplus, Softmax]
    learning_rate = 1e-3
    
    network = NeuralNetwork(len(layers), layers, activations, learning_rate)

    return network

if __name__ == "__main__":
    print("Loading data...")

    train_images, train_labels = load_data("mnist_train.csv")
    test_images, test_labels = load_data("mnist_test.csv")

    print("Data loaded.")

    network = None

    while(True):
        try:
            print("\n ----- NEURAL NETWORK ----- ")
            print(" 1. Train network ")
            print(" 2. Test network on full test dataset ")
            print(" 3. Test network on single image ")
            print(" 4. Save network ")
            print(" 5. Load network ")
            print(" 6. Exit ")
            option = int(input("\nEnter choice: "))
        except Exception as e:
            print(e)
            continue
        
        if option == 1:
            network = example_setup()
            if network != None:
                network.train(train_images, train_labels, CE_Loss)
            else:
                print("No network loaded.")

        elif option == 2:
            if network != None:
                network.test(test_images, test_labels, CE_Loss)
            else:
                print("No network loaded.")

        elif option == 3:
            if network != None:
                try:
                    index = int(input("Enter test data image index: "))
                    
                    if index < 0 or index >= 10000:
                        print("Error: Make sure index is between 0 and 9999.")

                    network.test_single(test_images[index], test_labels[index])
                except Exception as e:
                    print(e)
            else:
                print("No network loaded.")

        elif option == 4:
            if network != None:
                save_network(network)
            else:
                print("No network loaded.")

        elif option == 5:
            network = load_network()

        elif option == 6:
            exit(0)

        else:
            print(" Please enter a number between 1 and 6.")