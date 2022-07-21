import numpy as np

# each function from there requires data as np.ndarray
# tested only against y, y_pred being 1D arrays


def accuracy(y, y_pred):
    """Accuracy score"""
    correct = np.count_nonzero(np.all(y == y_pred, axis=1))
    return correct / len(y)

def confusion_matrix(y, y_pred, return_classes=False):
    """Calculates confusion matrix based on y and y_pred and returns it
    
    :param bool return_classes: wheather or not to return class names 
    :returns: conf_matrix if return_classes=False, otherwise (conf_matrix, classes)
    """
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
    """Mean squared error score"""
    return np.sum(np.abs(y_pred - y)) / len(y)

def root_mean_squared_error(y, y_pred):
    """Root mean squared error score"""
    return np.sqrt(np.sum((y - y_pred) ** 2) / len(y))

# how many of positive predictions are correct
def precision(y, y_pred):
    """Precision score, only for binary classifier"""
    tp = np.sum(np.logical_and(y == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y == 0, y_pred == 1))
    return tp / (fp + tp)

# how many of positive cases the classifier predicted corretly 
def recall(y, y_pred):
    """Recall score, only for binary classifier"""
    tp = np.sum(np.logical_and(y == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y == 1, y_pred == 0))
    return tp / (tp + fn)

def f1_score(y, y_pred):
    """F1 score, only for binary classifier"""
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    return 2 * p * r / (p + r)

def r_squared(y, y_pred):
    """R squared score"""
    mean = np.full(y.shape, np.mean(y))
    s1 = np.sum((y - y_pred) ** 2)
    s2 = np.sum((y - mean) ** 2)

    return (s2 - s1) / s2

