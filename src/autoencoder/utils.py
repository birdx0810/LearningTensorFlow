import json
import numpy as np

def get_batch(x, iteration, batch_size):
    time = [len(x[i][:,0]) for i in range(len(x))]

    start = iteration * batch_size
    end = start + batch_size

    x_mb = x[start:end]
    t_mb = time[start:end]

    return x_mb, t_mb

def get_anscombe_dataset():
    with open("./sample_data/anscombe.json", "r") as f:
        tmp = json.load(f)

    d1 = []
    d2 = []
    d3 = []
    d4 = []

    for d in tmp:
        if d['Series'] == 'I':
            d1.append([d["X"], d["Y"]])
        if d['Series'] == 'II':
            d2.append([d["X"], d["Y"]])
        if d['Series'] == 'III':
            d3.append([d["X"], d["Y"]])
        if d['Series'] == 'IV':
            d4.append([d["X"], d["Y"]])

    data = np.concatenate([[d1], [d2], [d3], [d4]])

    return data