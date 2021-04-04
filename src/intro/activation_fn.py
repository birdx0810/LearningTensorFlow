import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    """
    sigmoid(x) = 1/1+exp(-x)
    """
    return np.array(
        [
            1
            /
            (1+np.exp(-element))
            for element in X
        ]
    )

def tanh(X):
    """
    tanh(x) = \frac{exp(2x)-1}{exp(2x)+1 } 
    """
    return np.array(
        [
            (np.exp(2*element) - 1)
            /
            (np.exp(2*element) + 1)
            for element in X
        ]
    )

def relu(X):
    """
    relu(x) = max(0, x)
    """
    return np.array(
        [
            max(0.0, element)
            for element in X
        ]
    )

def softmax(X):
    """
    softmax(x) = exp(x_i)/sum_j(exp(x_j))
    """
    return np.array(
        [
            np.exp(element)
            /
            sum(np.exp(X))
        ]
    )
