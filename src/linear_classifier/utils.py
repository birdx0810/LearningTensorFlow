def get_batch(x, y, iteration, batch_size):
    start = iteration * batch_size
    end = start + batch_size

    x_mb = x[start:end]
    y_mb = y[start:end]
  
    return x_mb, y_mb