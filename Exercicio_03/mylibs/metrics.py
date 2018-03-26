import numpy as np
import math

def mse(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, Y_pred))

def mae(y_true, y_pred):
    return math.fabs((y_true - y_pred)) / len(y_true)
