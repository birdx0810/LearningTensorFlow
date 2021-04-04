from collections import Counter
import random

def pdf():
    """probability density function calculates the density of a continuous 
    random variable
    """
    pass

def cdf():
    """cumulative distribution function calculates the probability that a random 
    variable would be less than or equal to x for a given distribution
    """
    pass

def laplace_dist(mu, b):
    """a.k.a double exponential distribution
    Args:
        - mu: the location of the distribution
        - b: the scale parameter (diversity)
    """
    pass

def gaussian_dist(mu, sigma):
    """a.k.a. normal distribution
    Args:
        - mu: the mean of the distribution
        - sigma: the variance of the distribution
    """
    pass

def multinomial_dist(sample, prop):
    """multinomial distribution
    """
    total_prop = sum(prop)
    pdf = [p / total_prop for p in prop]

    cdf = [pdf[0]]
    for p in pdf[1:]:
        cdf.append(cdf[-1] + p)

    r = random.uniform(0, 1)
    for s, p in zip(sample, cdf):
        if r <= p:
            return s

if __name__ == "__main__":
    n_samples = 100000
    sample = ['a', 'b', 'c']
    prop = [2, 4, 5]

    res = []
    for _ in range(n_samples):
        res.append(multinomial_dist(sample, prop))

    c = Counter()
    c.update(res)

    out = {k : v / n_samples for k, v in c.items()}
    for k in sorted(out.keys()):
        print(f'{k}: {out[k]}')