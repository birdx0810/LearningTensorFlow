import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def mean_absolute_error(P, Y):
    """
    a.k.a. L1 Loss
    1/n \sum(|Y - P|)
    """
    return sum(
        np.abs(Y - P)
    )/len(Y)

def mean_square_error(P, Y):
    """
    a.k.a. L2 Loss
    1/n \sum((Y - P)^2)
    """
    return sum(
        (Y - P)**2
    )/len(Y)

def huber_loss(P, Y):
    """
    a.k.a. Smooth L1 Loss
    \cases{
        0.5(Y - P)^2, if |Y - P| < 1
        |Y - P| - 0.5, otherwise
    }
    """
    return sum(
        [
            0.5*(y - p)
            if (np.abs(y - p) < 1)
            else 
            np.abs(y - p) - 0.5
            for y, p in zip(P, Y)
        ]
    )

def cross_entropy(P, Y):
    """
    a.k.a. Negative Log Likelihood Loss
    -\sum(Y log(P))
    """
    eps = np.finfo(float).eps
    return -sum(
        Y * np.log(P + eps)
    )

def hinge_loss(D, Y, margin):
    """
    a.k.a. Pairwise Ranking Loss
    \cases{
        d, if y = 1
        max(0, margin - d) if y = -1
    }
    """
    return sum(
        [
            d
            if y == 1
            else
            max(0, margin - d)
            for d, y in zip(D, Y)
        ]
    )

def triplet_loss(A, P, N, margin, metric):
    """
    \max(0, m + d(A, P) - d(A, N))
    """
    loss = []
    for a, p, n in zip(A, P, N):
        if metric == "cosine_similarity":
            sim = cosine_similarity
            loss.append(
                max(0, margin - sim(a, p) + sim(a, n))
            )
        elif metric == "euclidean_distance":
            d = euclidean_distances
            loss.append(
                max(0, margin + d(a, p) - d(a, n))
            )
    return np.array(loss)/len(A)

def ranking_loss(Y, X_1, X_2, margin):
    """
    a.k.a. Triplet Loss
    \max(0, -y * x_1 - x_2)

    Args:
    - Y: rank relation
    - X_1: input 1
    - X_2: input 2
    - M: margin
    """
    loss = []
    for y, x_1, x_2 in zip (Y, X_1, X_2):
        loss.append(
            max(0, margin - y * (x_1 - x_2))        
        )
    return np.array(loss)/len(Y)
