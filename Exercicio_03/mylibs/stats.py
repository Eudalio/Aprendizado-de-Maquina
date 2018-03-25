import numpy as np
import math

def mean(x):
    return (np.sum(x)/len(x))

def var(y):
    media = mean(y)
    soma = 0;
    var = 0;
    for item in y:
        soma += ((item - media)**2)
    var = soma/len(y)
    return var;

def stdev(x):
    return math.sqrt(var(x))
    