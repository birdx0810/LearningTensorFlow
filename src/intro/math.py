import numpy as np

def log_sum_exp(X):
    """Log-Sum-Exponential for preventing floating underflow situations
    when calculating softmax.

    Original Softmax
    $$
    \sigma(z)_j 
    = \frac{e^{Z_j}}{\sum_{k=1}^K e^{z_k}}
    = \frac{e^{Z_j}}{e^{Z_1} + e^{Z_2} + ... + e^{Z_k+1}}
    $$

    LSE
    $$
    y = x_max + log(\sum_{i=1}^n e^{x_i-x_max})
    $$    
    """
    X_max = np.max(X)
    log_part = np.exp(X-X_max)
    Y = np.log(np.sum(log_part)) + X_max
    return Y