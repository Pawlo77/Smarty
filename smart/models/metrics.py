import numpy as np

from smart.errors import assertion
from smart.datasets import DataSet

# each function from there requires data as np.ndarray
# tested only against y, y_pred being 1D arrays


def accuracy(y, y_pred):
    correct = np.count_nonzero(y == y_pred)
    return correct / len(y)

def confusion_matrix(y, y_pred, return_classes=False):
    classes_ = np.unique(y)
    conf_ = np.zeros((len(classes_), len(classes_)), dtype=np.int32)

    lookup = {val: i for i, val in enumerate(classes_)}
    for y1, y2 in zip(y, y_pred):
        i = lookup[y1]
        j = lookup[y2]
        conf_[j, i] += 1

    if return_classes:
        return conf_, classes_
    return conf_

def mean_squared_error(y, y_pred):
    return np.sum(np.abs(y_pred - y)) / len(y)

def root_mean_squared_error(y, y_pred):
    return np.sqrt(np.sum((y - y_pred) ** 2) / len(y))

# only for binary classifier
# how many of positive predictions are correct
def precision(y, y_pred):
    tp = np.sum(np.logical_and(y == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y == 0, y_pred == 1))
    return tp / (fp + tp)

# only for binary classifier
# how many of positive cases the classifier predicted corretly 
def recall(y, y_pred):
    tp = np.sum(np.logical_and(y == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y == 1, y_pred == 0))
    return tp / (tp + fn)

# only for binary classifier
def f1_score(y, y_pred):
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    return 2 * p * r / (p + r)

def r_squared(y, y_pred):
    mean = np.full(y.shape, np.mean(y))
    s1 = np.sum((y - y_pred) ** 2)
    s2 = np.sum((y - mean) ** 2)

    return (s2 - s1) / s2

