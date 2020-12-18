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
    if metric == "cosine_similarity":
        sim = cosine_similarity
        return sum(
            max(0, margin - sim(A, P) + sim(A, N))
        )
    elif metric == "euclidean_distance":
        d = euclidean_distances
        return sum(
            max(0, margin + d(A, P) - d(A, N))
        )

def ranking_loss(A, P, N, M):
    """
    a.k.a. Triplet Loss

    Args:
    - A: anchor
    - P: positive
    - N: negative
    - M: margin
    """
    pass
