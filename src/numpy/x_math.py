import numpy as np

def log_sum_exp(X):
    Y = np.log(np.sum(np.exp(X)))