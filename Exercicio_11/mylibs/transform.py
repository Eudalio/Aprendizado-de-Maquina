import numpy as np

def normalize(x):
    x_norm = x.copy(x)
    n_cols = x.shape[1]
    for i in range(n_cols):
        x_norm[:, i] = (x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))
    return x_norm

def standardization(x):
    x_std = np.copy(x)
    n_cols = x.shape[1]
    for i in range(n_cols):
        x_std[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
    return x_std