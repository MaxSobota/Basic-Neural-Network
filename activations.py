import numpy as np

# Collection of various vectorized activation functions

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def softmax(x):
    stabilized = np.exp(x - np.max(x))
    return stabilized / stabilized.sum()