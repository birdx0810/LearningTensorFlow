"""
Reference:
https://github.com/scipy/scipy/blob/v1.5.4/scipy/spatial/distance.py
"""

import numpy as np
from scipy.special import softmax

def minowski_distance(X, Y, P):
    return np.linalg.norm(X - Y, ord=P)

def l1_distance(X, Y):
    """
    a.k.a. Manhatten distance
    """
    return np.abs(X - Y).sum()

def l2_distance(X, Y):
    """
    a.k.a. Euclidean distance
    """
    return minowski_distance(X, Y, P=2)

def cosine_similarity(X, Y):
    """
    X \cdot Y / |X| * |Y|
    """
    eps = np.finfo(float).eps
    return (
        (X @ Y.T)
        /
        (np.linalg.norm(X) * np.linalg.norm(Y) + eps)
    )

def attention(X, Y):
    """
    Attention weight matrix between elements of two vectors
    softmax((X @ Y.T)/\sqrt(len(X)))
    """

    numerator = X @ Y.T
    denominator = np.sqrt(X.shape[-1])
    return (
        softmax(numerator/denominator) # 512 * 10 
    )

