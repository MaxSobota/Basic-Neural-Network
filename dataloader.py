import csv
import numpy as np
import pickle
from datetime import datetime

# Loads CSV data into one-hot labels and 28x28 numpy arrays
def load_data(filename):
    print(f"Loading data from {filename}...")
    
    with open(filename, newline="") as file:
        reader = csv.reader(file)
        
        image_values = []
        labels = []

        for row in reader:
            values = [int(x) for x in row]

            # Creating one-hot vector for each label
            label = np.zeros((10, ), dtype=np.int64)
            label[values[0]] = 1
            labels.append(label)

            image_values.append(values[1:]) 

    images = [np.array(img, dtype=np.float64).reshape(28, 28) for img in image_values]

    print("Data loaded.")

    return images, labels

# Saves networks with a unique filename based on the time
def save_network(network):
    time_string = "network-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Saving most recently trained network to {time_string}.pkl...")

    try:
        with open(f"{time_string}.pkl", "wb") as file:
            pickle.dump(network, file)
    except Exception as e:
        print(e)

# Check that the .pkl file is a NeuralNetwork object
def load_network(filename):
    try:
        with open(filename, "rb") as file:
            network = pickle.load(file)

        return network
    except Exception as e:
        print(e)
        return None
    
# Helper class to load data into GUI
class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    # Len of labels == len of images
    def __len__(self):
        return len(self.images)

    def get(self, index):
        return self.images[index], self.labels[index]
