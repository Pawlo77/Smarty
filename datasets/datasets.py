import numpy as np
import requests

from errors import assertion
from .training_properties import TrainingProperty
from .stats_properties import StatisticsProperty


INT_TYPES = (
    np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64
)
FLOAT_TYPES = (
    np.float16, np.float32, np.float64,
)
# enables / disables auto detection, administrated by DummyWriter
_AUTO_DETECTION = True


class DataCol:
    def create(col, dtype=None):
        if isinstance(col, np.ndarray):
            if dtype is not None: # if any dtype passed 
                return col.astype(dtype)
            if _AUTO_DETECTION: # if auto detection is enabled
                return DataCol._find_type(col)
            return col
        if dtype is not None:
            return np.array(col, dtype=dtype) # convert to np.ndarray with specified dtype
        return DataCol._find_type(np.array(col)) # convert to np.ndarray and find dtype

    def _find_type(col): # note: it doesn't check if float precision meet, for floats preferable to do it alone
        def check(info, min_, max_):
            return info.min <= min_ and info.max >= max_

        try:
            col = col.astype("f8")
        except Exception: # it's not numerical, keep what numpy picked
            return col
        else:
            min_, max_ = np.min(col), np.max(col)

            if np.all(col % 1 == 0): # if integer
                for dtype in INT_TYPES:
                    info = np.iinfo(dtype)
                    if check(info, min_, max_): # if in defined range
                        return col.astype(dtype)
            else:
                for dtype in FLOAT_TYPES: # float
                    info = np.finfo(dtype)
                    if check(info, min_, max_): # if in defined range
                        return col.astype(dtype) 


# allows to use DataCol without auto-dtype detecion
class DummyWriter:
    def __init__(self, root):
        self.root = root

    def __enter__(self, *args): # turn off auto dtype detection on entry
        _AUTO_DETECTION = False

    def __exit__(self, *args): # turn on auto dtype detection on exit
        _AUTO_DETECTION = True


class DataSet(TrainingProperty, StatisticsProperty):

    def __init__(self):
        super().__init__()

        self.matrix_ = None
        self.columns_ = [] # to hold column names
        self.dtypes_ = []
        
        # turn on/off auto dtype detection by calling with ds.dummy_writer: ...
        self.dummy_writer = DummyWriter(self) 

    def get_shape_(self):
        if self.matrix_ is None or not self.matrix_.any():
            return (0, )
        return self.matrix_.shape

    # return categorical columns idxs
    def categorical_idxs_(self): 
        idxs = []
        for i in range(len(self.dtypes_)):
            if self._is_categorical(i):
                idxs.append(i)
        return idxs

    # return numerical columns idxs
    def numerical_idxs_(self):
        idxs = []
        for i in range(len(self.dtypes_)):
            if not self._is_categorical(i):
                idxs.append(i)
        return idxs

    def empty_(self):
        return self.get_shape_() == (0, )

    # returns non-numerical columns
    def get_catedorical(self):
        return self.matrix_[:, self.categorical_idxs_()]

    # returns numerical columns
    def get_numerical(self):
        return self.matrix_[:, self.numerical_idxs_()]
    
    # accepts a 2D np.ndarray or a 2D nested python list
    # dtypes - one for entire matrix or none (auto-detecion) or one for each column
    # columns - names for all columns or none (auto-naming)
    def from_object(self, matrix, columns=[], dtypes=[]):
        # if one dtype provided for entire array
        if not isinstance(dtypes, list) and not isinstance(dtypes, np.ndarray):
            dtypes = [dtypes] * len(matrix[0])

        assertion(isinstance(matrix, list | np.ndarray) and isinstance(matrix[0], list | np.ndarray), "Wrong data format provided")
        assertion(len(matrix) != 0 and len(matrix[0]) != 0, "Empty sequence.")
        assertion(len(dtypes) == 0 or len(dtypes) == len(matrix[0]), "Dtypes don't match number of columns in matrix.")
        assertion(len(columns) == 0 or len(columns) == len(matrix[0]), "Columns don't match number of columns in matrix'")

        cols = []
        for i in range(len(matrix[0])):
            dtype = dtypes[i] if len(dtypes) else None
            name = columns[i] if len(columns) else str(i)

            if isinstance(matrix, list):
                col = DataCol.create(
                    [matrix[j][i] for j in range(len(matrix))], dtype=dtype
                ).reshape(-1, 1)
            else:
                col = DataCol.create(
                    matrix[:, i], dtype=dtype
                ).reshape(-1, 1)
            
            self.dtypes_.append(col.dtype)
            self.columns_.append(name)
            cols.append(col)

        self.columns_ = np.array(self.columns_, dtype="U32")
        self.dtypes_ = np.array(self.dtypes_, dtype="object")
        self.matrix_ = np.concatenate(cols, axis=1, dtype="object")
        return self

    # return data - matrix_ np.ndarray
    def numpy(self):
        return self.matrix_

    # removes columns from DataSet and returns them as new DataSet instance
    # key - column name, list of columns names, column idx or list of columns idxs
    def drop_c(self, key):
        if isinstance(key, str): # single column by name
            key = self._get_idx(key)
        elif isinstance(key, list) and isinstance(key[0], str): # multiple columns by names
            key = [self._get_idx(k) for k in key]
        assertion(type(key) == int or (type(key) == list and type(key[0]) == int) or type(key) == slice, "Wrong column specifier")

        dt = self.matrix_[:, key]
        c = self.columns_[key]
        d = self.dtypes_[key]

        # if it was a single column
        if len(dt.shape) == 1:
            dt = dt.reshape(-1, 1)
            c = [c]
            d = [d]

        self.matrix_ = np.delete(self.matrix_, key, axis=1)
        self.dtypes_ = np.delete(self.dtypes_, key)
        self.columns_ = np.delete(self.columns_, key)

        return DataSet().from_object(dt, columns=c, dtypes=d)
        
    # removes rows from dataset and returns them as new DataSet instance
    # key - row idx or list of rows idxs
    def drop_r(self, key):
        assertion(type(key) == int or (type(key) == list and type(key[0]) == int) or type(key) == slice, "Wrong row specifier")

        dt = self.matrix_[key, :]
        self.matrix_ = np.delete(self.matrix_, key, axis=0)
        return DataSet().from_object(dt, columns=self.columns_, dtypes=self.dtypes_)

    # add colum/s
    # c - list np.ndarray for 1 column, list of lists or 2D np.ndarray for multiple columns
    # columns - name for all columns or none (auto-naming)
    # dtypes - none, one dtype for all or one dtype for each column
    # pos - none (works as append), int - add all continously at that position, list of ints - pos for each column
    def add_c(self, c, columns=None, dtypes=None, pos=-1):
        assertion(isinstance(c, list | np.ndarray), "Wrong column type provided.")
        num_cols = len(c[0]) if isinstance(c[0], list | np.ndarray) else 1

        if not isinstance(dtypes, list | np.ndarray): # one type for all (None included)
            dtypes = [dtypes] * num_cols 
        if not isinstance(columns, list | np.ndarray):
            columns = [columns] * num_cols
        assertion(len(dtypes) == num_cols, "Number of provided dtypes don't match number of columns")
        assertion(len(columns) == num_cols, "Number of provided names don't match number of columns")

        if num_cols == 1: # single column as list or np.ndarray
            columns[0] = columns[0] if columns[0] else self._get_name()
            c = DataCol.create(c, dtype=dtypes[0]).reshape(-1, 1)
            dtypes[0] = c.dtype
        else: # multiple columns
            cols = []
            for i in range(num_cols):
                columns[i] = columns[i] if columns[i] is not None else self._get_name()

                if isinstance(c[0], list):
                    col = DataCol.create(
                        [c[j][i] for j in range(len(c))],
                        dtype=dtypes[i],
                    ).reshape(-1, 1)
                else:
                    col = DataCol.create(
                        c[:, i], 
                        dtype=dtypes[i],
                    ).reshape(-1, 1)

                dtypes[i] = col.dtype
                cols.append(col)
            
            c = np.concatenate(cols, axis=1, dtype="object")
        columns = np.array(columns, dtype="U32")
        dtypes = np.array(dtypes, dtype="object")

        if pos == -1: # works as append
            pos = [self.get_shape_()[1]] * num_cols
        elif not isinstance(pos, list | np.ndarray): # insert as one at given index
            pos = [pos] * num_cols
        assertion(len(pos) == num_cols, "Number of provided positions don't match number of columns")

        self.matrix_ = np.insert(self.matrix_, pos, c, axis=1)
        self.dtypes_ = np.insert(self.dtypes_, pos, dtypes)
        self.columns_ = np.insert(self.columns_, pos, columns)

    # r - list np.ndarray for one row, list of lists or np.ndarray for multiple rows
    # pos - none (works as append), int - add all continously at that position, list of ints - pos for each column
    def add_r(self, r, pos=-1):
        assertion(isinstance(r, list | np.ndarray), "Wrong row type provided.")
        if not isinstance(r[0], list | np.ndarray): # single row
            r = np.array([r], dtype="object")
        else: # multiple rows
            r = np.array(r, dtype="object")
    
        if pos == -1: # works as append
            pos = [self.get_shape_()[0]] * len(r)
        elif not isinstance(pos, list | np.ndarray): # insert as one at given index
            pos = [pos] * len(r)
        assertion(len(pos) == len(r), "Number of provided positions don't match number of rows")
        
        # convert to columns dtypes
        for i in range(self.get_shape_()[1]):
            r[:, i] = r[:, i].astype(self.dtypes_[i])

        self.matrix_ = np.insert(self.matrix_, pos, r, axis=0)

    # joins 2 DataSets together
    # pos like in self.add_c or self.add_r
    # axis -> 1 - join column-wise, 0 - join row-wise
    def join(self, dt, pos=-1, axis=1):
        if axis == 1:
            self.add_c(dt.matrix_, columns=dt.columns_, dtypes=dt.dtypes_, pos=pos)
        elif axis == 0:
            self.add_r(dt.matrix_, pos=pos)

    # returns copy of itself
    def copy(self): 
        new_ds = DataSet().from_object(self.matrix_, columns=self.columns_, dtypes=self.dtypes_)
        super().copy(new_ds)
        return new_ds

    # returns idx of first column named name
    def _get_idx(self, name):
        assertion(name in self.columns_, "Column not found.")
        return int(np.where(self.columns_ == name)[0][0])

    # checks if column idx is categorical or numerical
    def _is_categorical(self, idx):
        return not self.dtypes_[idx] in INT_TYPES + FLOAT_TYPES

    # generate new column name if it wasn't provided (util func for add_c)
    def _get_name(self):
        i = 0
        while True:
            if str(i) not in self.columns_:
                return str(i)
            i += 1

    # key - cannot by numpy
    def __getitem__(self, key):
        if isinstance(key, str): # single column by name
            key = self._get_idx(key)
        if isinstance(key, int): # single column by idx
            return DataCol.create(self.matrix_[:, key], dtype=self.dtypes_[key]) # return as DataCol

        if isinstance(key, list | np.ndarray) and isinstance(key[0], str): # multiple columns by names
            key = [self._get_idx(k) for k in key]
        if isinstance(key, list | np.ndarray) and isinstance(key[0], int): # multiple columns by idxs
            return self.matrix_[:, key]

        if isinstance(key[1], str): # some rows and column by name
            key = (key[0], self._get_idx(key[1]))
        if isinstance(key[1], int): # some rows and column by idx
            return DataCol.create(self.matrix_[key], dtype=self.dtypes_[key[1]]) # return as DataCol
        
        if isinstance(key[1], list | np.ndarray) and isinstance(key[1][0], str): # some rows and columns by names
            key = (key[0], [self._get_idx(k) for k in key[1]])
        return self.matrix_[key]

    # key - cannot by numpy
    def __setitem__(self, key, value):
        if isinstance(key, str): # single column by name
            key = (slice(self.get_shape_()[0]), [self._get_idx(key)])
        elif isinstance(key, list | np.ndarray) and isinstance(key[0], str): # multiple columns by names
            key = [self._get_idx(k) for k in key]
            key = (slice(self.get_shape_()[0]), key)

        elif isinstance(key, int): # single column by idx
            key = (slice(self.get_shape_()[0]), [key])
        elif isinstance(key, list | np.ndarray) and isinstance(key[0], int): # multiple columns by idxs
            key = (slice(self.get_shape_()[0]), key)
            

        elif isinstance(key[1], str): # some rows and column by name
            key = (key[0], self._get_idx(key[1]))
        elif isinstance(key[1], list | np.ndarray) and isinstance(key[1][0], str): # some rows and columns by names
            key = (key[0], [self._get_idx(k) for k in key[1]])
        
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype="object")

        if len(value.shape) == 1:
            value = value.reshape(-1, 1)

        # if entire column/s reassigned, update its dtype in self.dtype_
        if value.shape[0] == self.get_shape_()[0]: 
            cols = []
            for i in range(value.shape[1]):
                dtype = self.dtypes_[key[1][i]]
                col = DataCol.create(
                    value[:, i],
                    dtype=dtype,
                ).reshape(-1, 1)

                self.dtypes_[key[1][i]] = col.dtype
                cols.append(col)
            value = np.concatenate(cols, axis=1, dtype="object")

        # cast each entry to desired type
        else:
            dtypes_ = self.dtypes_[key[1]]
            if not isinstance(dtypes_, np.ndarray): # for one-column
                dtypes_ = np.array([dtypes_])

            for i in range(value.shape[1]):
                value[:, i] = value[:, i].astype(dtypes_[i])

        self.matrix_[key] = value.reshape(self.matrix_[key].shape)

    def __repr__(self):
        self.head()
        return "\r"

    def __str__(self):
        self.head()
        return "\r"

    def __len__(self):
        return len(self.matrix_) if self.matrix_ is not None else 0
