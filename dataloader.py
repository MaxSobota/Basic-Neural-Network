import csv
import numpy as np

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
