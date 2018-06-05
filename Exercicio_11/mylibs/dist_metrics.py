import numpy as np

def minkowski_distance(X, row, p=2):
    X_ = np.abs(X - row) ** p
    return np.sum(X_, axis=1) ** (1/p)

def euclidean_distance(X, row):
    return minkowski_distance(X, row)
    
    #X_ = (X - row) ** 2
    #return np.sum(X_, axis=1) ** 0.5

#def euclidean_distance(X, row):
    #np.sqrt(np.sum((X - row) ** 2))
    #return minkowski_distance(X, row)

def manhattan_distance(X, row):
    return minkowski_distance(X, row, 1)

def chebyshev_distance(X, row):
    return np.max(np.abs(X - row))

