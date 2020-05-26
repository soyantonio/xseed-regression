import numpy as np

def descent(attributes, params, ys, learning_rate=0.00001):
    m = ys.size
    predicted = np.dot(attributes, params)
    params = params - np.dot(attributes.T, predicted - ys)*learning_rate/m
    return params