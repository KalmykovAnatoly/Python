import numpy as np


def norm(vector):
    e = np.array(vector)
    return e / np.linalg.norm(e)


def projection(a, b):
    return np.dot(a, b) / np.dot(b, b)
