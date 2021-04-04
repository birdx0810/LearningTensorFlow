import numpy as np

# Set random seed
np.random.seed = 42

# y = w * x + b
x = np.array([5])
y = 3 * x + 2
w = np.random.rand(1, 1)
b = np.random.rand(1)

def mse_loss(w, b, x, y):
    return sum((w * x + b - y) ** 2) / len(x)

def backpropagation(w, b, x, y):
    dL_dw = sum((w * x + b - y)* x) / len(x) * 2
    dL_db = sum(w * x + b - y) / len(x) * 2

    return dL_dw, dL_db

def optimization(w, b, dL_dw, dL_db, lr):
    w = w - lr * dL_dw
    b = b - lr * dL_db

    return w, b

epochs = 10000
learning_rate = 1e-5

for epoch in range(epochs):
    print(f"Loss: {mse_loss(w, b, x, y).item()}")
    dL_dw, dL_db = backpropagation(w, b, x, y)
    w, b = optimization(w, b, dL_dw, dL_db, learning_rate)

print(f"y: {y.item()}\tx: {x.item()}")
print(f"W: 3\tb: 2")
print(f"y = {round(w.item(), 3)} x + {round(b.item(), 3)}")
