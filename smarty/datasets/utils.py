import numpy as np
import requests

from .datasets import DataSet
from smarty.errors import assertion


def load_data(filename="", url="", header=True, sep=","):
    """Loads data from given source and returns a DataSet object

    :param str | Path filename: filename / path to source file
    :param str url: direct link to source file, note: file should be open to downloads without any logins or captcha
    :param bool header: True - furst line of file will be treated as header (columns names will be generated based on it)
    :param str sep: seperator
    :raises: AssertionError if no source provided
    """
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


def train_test_split(ds, split_ratio=0.8, shuffle=True, seed=42, *args, **kwargs):
    """Splits ds into 2 seperate DataSets - one for training and one for testing

    :param DataSet | np.ndarray ds: a DataSet to be splited
    :param float between (0.0, 1.0) split_ratio: the split ratio (how many of the ds is to be put to a training set)
    :param bool shuffle: if True random rows are being shuffled, False - first split_ratio * 100% of the ds will be put in the training set
    :param int seed: random seed for numpy (used only with shuffle=True)
    :returns: training_ds, test_ds (both instances of DataSet)
    """
    assertion(0.0 < split_ratio < 1.0 and isinstance(shuffle, bool) and isinstance(seed, int) and isinstance(ds, DataSet | np.ndarray), "Wrong entry data provided")

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

def cross_val_split(ds, folds=3, shuffle=True, seed=42, drop_reminder=False, *args, **kwargs):
    """Splits ds into folds seperate DataSets

    :param DataSet | np.ndarray ds: a DataSet to be splited
    :param int >= 2 folds: how many folds to be created, has to be smaller than number of rows in the ds
    :param bool shuffle: if True random rows are being shuffled, False - first fold uses n first rows, second n to 2n etc
    :param int seed: random seed for numpy (used only with shuffle=True)
    :param bool drop_reminder: if True last fold is skipped if not full
    :returns: list of DataSets, each one being seperate fold
    """
    assertion(insinstance(folds, int) and isinstance(shuffle, bool) and insinstance(seed, int) and isinstance(drop_reminder, bool), "Wrong entry data provided")
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