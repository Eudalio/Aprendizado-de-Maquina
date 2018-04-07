import numpy as np
from sklearn import metrics
import math

def mse(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return sum(np.abs((y_true - y_pred))) / len(y_true)

def accuracy(y_true, y_pred):
    matrix_confusion = metrics.confusion_matrix(y_true, y_pred)
    return sum(np.diagonal(matrix_confusion))/len(y_true)

def precision(y_true, y_pred):
    mc = metrics.confusion_matrix(y_true, y_pred)
    return mc[0,0] / sum(mc[:,0])

def recall(y_true, y_pred):
    mc = metrics.confusion_matrix(y_true, y_pred)
    return mc[0,0] / sum(mc[0,:])

def f1_measure(y_true, y_pred):
    mc = metrics.confusion_matrix(y_true, y_pred)
    return (2 * ((precision(y_true, y_pred) * recall(y_true, y_pred)) / (precision(y_true, y_pred) + recall(y_true, y_pred))))