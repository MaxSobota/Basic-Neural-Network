import csv
import numpy as np

# Loads CSV data into labels and 28x28 numpy arrays
def load_data():
    with open("mnist_train.csv", newline="") as file:
        reader = csv.reader(file)
        
        image_values = []
        labels = []

        for row in reader:
            values = [int(x) for x in row]
            labels.append(values[0])
            image_values.append(values[1:]) 

    images = [np.array(img).reshape(28, 28) for img in image_values]

    return images, labels
