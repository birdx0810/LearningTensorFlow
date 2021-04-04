# Example input vector for plotting functions
X = np.arange(start=-10, stop=10, step=0.2)

# Function for graph plot using matplotlib
def plot_linear(X):
    Y = np.linspace(start=-50, stop=50, num=100)
    plt.plot(X, Y)
    plt.title("Linear Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

def plot_sigmoid(X):
    S = sigmoid(X)
    plt.plot(X, S)
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid()
    plt.show()

def plot_tanh(X):
    T = tanh(X)
    plt.plot(X, T)
    plt.title("tanh Function")
    plt.xlabel("x")
    plt.ylabel("tanh(x)")
    plt.grid()
    plt.show()

def plot_relu(X):
    R = relu(X)
    plt.plot(X, R)
    plt.title("ReLU Function")
    plt.xlabel("x")
    plt.ylabel("relu(x)")
    plt.grid()
    plt.show()

def plot_log():
    X = np.arange(0.01, 1, 0.01)
    Y = -np.log(X)
    plt.plot(X, Y)
    plt.title("Negative Log Likelihood")
    plt.xlabel("x")
    plt.ylabel("-log(x)")
    plt.grid()
    plt.show()
