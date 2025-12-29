import csv
import numpy as np
import pickle
from datetime import datetime

# Loads CSV data into one-hot labels and 28x28 numpy arrays
def load_data(filename):
    with open(filename, newline="") as file:
        reader = csv.reader(file)
        
        image_values = []
        labels = []

        for row in reader:
            values = [int(x) for x in row]

            # Creating one-hot vector for each label
            label = np.zeros((10, ))
            label[values[0]] = 1
            labels.append(label)

            image_values.append(values[1:]) 

    # Normalize to between 0 and 1
    images = [(np.array(img).reshape(28, 28) / 255) for img in image_values]

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

def load_network():
    filename = input("Enter filepath (without .pkl extension): ")

    try:
        with open(f"{filename}.pkl", "rb") as file:
            network = pickle.load(file)
        return network
    except Exception as e:
        print(e)
        return None