import numpy as np

class ReLU:
    def func(x):
        return np.maximum(0, x)

    def func_derivative(x):
        return np.greater(x, 0)
    
class Softplus:
    def func(x):
        return np.log(1 + np.exp(x))

    def func_derivative(x):
        return 1 / (1 + np.exp(-x))

class Softmax:
    def func(x):
        stabilized = np.exp(x - np.max(x))
        return stabilized / stabilized.sum()
    
class CE_Loss:
    def func(prediction, label):
        epsilon = 1e-12
        prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
        return -np.sum(label * np.log(prediction))

    def func_derivative(prediction, label):
        return prediction - label