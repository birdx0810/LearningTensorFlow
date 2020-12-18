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

# Example input vector for plotting functions
X = np.arange(start=-10, stop=10, step=0.2)

# Function for graph plot using matplotlib
def plot_linear(X, Y):
    Y = np.linspace(start=-50, stop=50, num=100)
    plt.plot(X, Y)
    plt.title("Linear Function")
    plt.grid()
    plt.show()

def plot_sigmoid(X, S):
    S = sigmoid(X)
    plt.plot(X, S)
    plt.title("Sigmoid Function")
    plt.grid()
    plt.show()

def plot_tanh(X):
    T = tanh(X)
    plt.plot(X, T)
    plt.title("tanh Function")
    plt.grid()
    plt.show()

def plot_relu(X):
    R = relu(X)
    plt.plot(X, R)
    plt.title("ReLU Function")
    plt.grid()
    plt.show()
