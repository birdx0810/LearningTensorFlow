from collections import Counter
import random

def multinomial_dist(sample, prop):
    total_prop = sum(prop)
    pdf = [p / total_prop for p in prop]

    cdf = [pdf[0]]
    for p in pdf[1:]:
        cdf.append(cdf[-1] + p)

    r = random.uniform(0, 1)
    for s, p in zip(sample, cdf):
        if r <= p:
            return s

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