import numpy as np
import requests

from .datasets import DataSet
from smart.errors import assertion


# loads datafile from disc / url link and returns DataSet object
# add handling missing data here
def load_data(filename="", url="", header=True, sep=","):
    assertion(filename or url, "Provide data source.")

    if filename: # use local file
        file = open(filename, "r")
        lines = file.readlines()
    else: # use url
        file = requests.get(url, stream=True).content
        lines = file.decode("utf8").strip().split("\n")

    for line_idx in range(len(lines)):
        lines[line_idx] = lines[line_idx].strip().split(sep)
    if header:
        header = lines[0]
        lines = lines[1:]
    
    if not header:
        header = []
    return DataSet().from_object(lines, columns=header) 

# accepts np.ndarray or DataSet
def train_test_split(ds, split_ratio=0.8, shuffle=True, seed=42, *args, **kwargs):
    indeces = np.arange(len(ds))
    train_size = int(np.floor(split_ratio * len(ds)))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indeces)

    kw = {}
    if isinstance(ds, DataSet):
        kw["columns"] = ds.columns_
        kw["dtypes"] = ds.dtypes_
        data = ds.numpy()
    else:
        data = ds
    
    train_set = DataSet().from_object(
        data[indeces[:train_size], :], **kw
    )
    test_set = DataSet().from_object(
        data[indeces[train_size:], :], **kw
    )

    ds.train_copy(train_set)
    ds.train_copy(test_set)
    return train_set, test_set

# accepts np.ndarray or DataSet
def cross_val_split(ds, folds=3, shuffle=True, seed=42, drop_reminder=False, *args, **kwargs):
    assertion(folds >= 2, "Minimum number of folds is 2.")
    assertion(len(ds) > folds, "Number of folds exceeds length of dataset")
    indeces = np.arange(len(ds))
    fold_size = len(ds) // folds

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indeces)

    kw = {}
    if isinstance(ds, DataSet):
        kw["columns"] = ds.columns_
        kw["dtypes"] = ds.dtypes_
        data = ds.numpy()
    else:
        data = ds

    splits = []
    s = 0
    while s + fold_size <= len(ds):
        idxs = indeces[s:s + fold_size]
        s += fold_size

        splits.append(DataSet().from_object(
            data[idxs, :], **kw
        )) 

    if not drop_reminder and s != len(data):
        idxs = indeces[s:]
        splits.append(DataSet().from_object(
            data[idxs, :], **kw
        ))

    for idx in range(len(splits)):
        ds.train_copy(splits[idx])
    return splits