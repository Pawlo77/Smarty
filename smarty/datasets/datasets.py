from tkinter import N
import numpy as np

from smarty.errors import assertion
from smarty.config import get_config, set_config
from .training_properties import TrainingProperty
from .stats_properties import StatisticsProperty


INT_TYPES = (
    np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64
)
"""List of int dtypes auto-detection will check, each one being a native np.dtype"""

# dropped float16 whus of its low-accuracy
# FLOAT_TYPES = (
#     np.float16, np.float32, np.float64,
# )
FLOAT_TYPES = (
    np.float32, np.float64,
)
"""List of float dtypes auto-detection will check, each one being a native np.dtype"""


# helper class for DataSet
class DataCol:
    def create(col, dtype=None):
        if isinstance(col, np.ndarray):
            col = DataCol._handle_empty(col)

            if dtype is not None: # if any dtype passed 
                return col.astype(dtype)
            if get_config("_AUTO_DETECTION"): # if auto detection is enabled
                return DataCol._find_type(col)
            return col

        col = np.array(col)
        col = DataCol._handle_empty(col)
        if dtype is not None:
            return col.astype(dtype) # convert to np.ndarray with specified dtype
        return DataCol._find_type(col) # convert to np.ndarray and find dtype

    def _handle_empty(col):
        empty = np.where(col=='')[0]
        if len(empty) != 0: # fill empty cells
            try:
                col = col.astype("f")
            except Exception as e:
                col = col.astype("U")
                col[empty] = "nan" 
            else:
                col[empty] = np.nan
        return col

    def _find_type(col): # note: it doesn't check if float precision meet, for floats preferable to do it alone
        def check(info, min_, max_):
            return info.min <= min_ and info.max >= max_

        try:
            col = col.astype("f8")
        except Exception: # it's not numerical, keep what numpy picked
            return col
        else:
            min_, max_ = np.nanmin(col), np.nanmax(col)
            assertion(min_ != "nan" and max_ != "nan", "Entire column empty")

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
        set_config("_AUTO_DETECTION", False)

    def __exit__(self, *args): # turn on auto dtype detection on exit
        set_config("_AUTO_DETECTION", True)


class DataSet(TrainingProperty, StatisticsProperty):
    """Object holding the accual data, allowing to name its columns and choose their dtypes. It's data can be set and accessed via [] (like python's list) - see below example

    :var np.ndarray matrix\_: array holidng the data (columns - its 1 axis), preferable to get it via .numpy() method
    :var np.ndarray columns\_: at each position i holds name of ith column of the dataset (if not provided smallest not used number will be picked for name)
    :var np.ndarray dtypes\_: at each position i holds dtype of ith column of the dataset
    :var dummy_writer: allows to write columns to the dataset without changing their dtype, ex:

    .. code-block:: python
        :linenos:
    
        from smarty.datasets import load_data
        import numpy as np

        ds = load_data("iris.txt")
        # here ds["species"] will remain as dtype U, not uint8
        with ds.dummy_writer:
            ds["species"] = np.ones((150, 1)) # overwrite entire "species" column

        print(ds[10, "petal_length"]) # 11th row of PentalLength column
        print(ds[1]) # entire 2nd column
        print(ds[11, :]) # entire 12th row
        print(ds[12, 3]) # 4th column, 13th row
        print(ds["sepal_width"]) # entire SepalWidth column
        ds[41, "sepal_width"] = 2.3 # set 42th row of "sepal_width" column to 2.3
        ds[36, 3] = 1.7 # set 37th row of 4th column to 1.7

    .. warning::
        Providing columns with missing data (ie "") might cause them to convert to dtype of "U" or float regardless to your actions
    """

    def __init__(self):
        super().__init__()

        self.matrix_ = None
        self.columns_ = [] # to hold column names
        self.dtypes_ = []
        
        # turn on/off auto dtype detection by calling with ds.dummy_writer: ...
        self.dummy_writer = DummyWriter(self) 

    def get_shape_(self):
        """
        :returns: matrix\_ shape
        """
        return self.matrix_.shape

    def categorical_idxs_(self): 
        """
        :returns: list of indexes of categorical columns in the dataset.
        """
        idxs = []
        for i in range(len(self.dtypes_)):
            if self._is_categorical(i):
                idxs.append(i)
        return idxs

    def numerical_idxs_(self):
        """
        :returns: list of indexes of numerical columns in the dataset.
        """
        idxs = []
        for i in range(len(self.dtypes_)):
            if not self._is_categorical(i):
                idxs.append(i)
        return idxs

    def empty_(self):
        """
        :returns: boolean indicating whether the dataset is empty.
        """
        return self.get_shape_() == (0, ) or self.get_shape_() == (1, 0)

    def get_catedorical(self):
        """
        :returns: np.ndarray holding all dataset's categorical columns.
        """
        return self.matrix_[:, self.categorical_idxs_()]

    def get_numerical(self):
        """ 
        :returns: np.ndarray holding all dataset's numerical columns.
        """
        return self.matrix_[:, self.numerical_idxs_()]
    
    # accepts a 2D np.ndarray or a 2D nested python list
    # dtypes - one for entire matrix or none (auto-detecion) or one for each column
    # columns - names for all columns or none (auto-naming)
    def from_object(self, matrix, columns=[], dtypes=[]):
        """Creates and returns DataSet object
        
        :param np.ndarray | 2D nested list matrix: matrix containing the data
        :param np.ndarray | list columns: list containing column name for each column in the matrix (not required)
        :param np.ndarray | list | dtype dtypes: list containing dtype name for each column in the matrix (not required) or one single dtype (like str) that will be used for all columns
        :raises: AssertionError if matrix is empty
        """
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

    def numpy(self):
        """
        :returns: DataSet pure data (underlying matrix - np.ndarray)
        """
        return self.matrix_

    def drop_c(self, key, delete=True):
        """Removes columns from DataSet and returns them as new DataSet instance

        :param str | list | int key: column name (str), list of column names (list of strs), column index (int) or list of columns indexes (list of ints)
        :param bool delete: If false value rows won't be deleted but only returned
        :raises: AssertionError if key is not valid
        """
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

        if delete:
            self.matrix_ = np.delete(self.matrix_, key, axis=1)
            self.dtypes_ = np.delete(self.dtypes_, key)
            self.columns_ = np.delete(self.columns_, key)

        return DataSet().from_object(dt, columns=c, dtypes=d)
        
    def drop_r(self, key, delete=True):
        """Removes rows from dataset and returns them as new DataSet instance
        
        :param int | list key: row idx or list of rows indexes (list of ints)
        :param bool delete: If false value rows won't be deleted but only returned
        :raises: AssertionError if key is not valid
        """
        assertion(type(key) == int or (type(key) == list and type(key[0]) == int) or type(key) == slice, "Wrong row specifier")

        dt = self.matrix_[key, :]

        if delete:
            self.matrix_ = np.delete(self.matrix_, key, axis=0)

        if len(dt.shape) == 1: # single row
            dt = dt[np.newaxis, ...]

        new = DataSet().from_object(dt, columns=self.columns_, dtypes=self.dtypes_)
        self.train_copy(new)
        return new

    def add_c(self, c, columns=None, dtypes=None, pos=-1):
        """Adds column/s to a dataset

        :param list | np.ndarray c: list or 1D np.ndarray for one column, list of list or 2D np.ndarray for multiple columns to be added
        :param list of str | None columns: list - name for each column in c or None (auto naming with lowest possible number)
        :param list | None | any dtype dtypes: None for auto dtype detection, any dtype that will be used for all columns in c, list of dtypes - one for each column in c
        :param int | list | np.ndarray pos: position where to put new rows. ***defaults to -1* - works like append**, int to insert rows as one block, list of ints - for each row its desired location

        .. warning::
            providing dtypes=None with disabled auto dtype detection will cause usage of dtype choosen by numpy (in case of fed in list) or same as in fed in np.ndarray
        """

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

    def add_r(self, r, pos=-1):
        """Adds row/s to a dataset

        :param list | np.ndarray r: list or 1D np.ndarray for one row, list of list or 2D np.ndarray for multiple rows to be added
        :param int | list | np.ndarray pos: position where to put new rows. ***defaults to -1* - works like append**, int to insert rows as one block, list of ints - for each row its desired location

        .. warning::
            Each row must be same length as number of columns in the dataset. DataSet will try to cast each value in the row to corresponding column's dtype, which might change "0001" to 1 or cause an error (ex trying to case "tree" to uin8)
        """
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

    def join(self, dt, pos=-1, axis=1):
        """Joins DataSet dt to itself

        :param int | list | np.ndarray pos: position where to put new rows. ***defaults to -1* - works like append**, int to insert rows as one block, list of ints - for each row its desired location
        :param int axis: 1 for column-wise join, 0 for row-wise join
        """
        assertion(isinstance(dt, DataSet), "dt must be a smarty.datasets.DataSet")
        assertion(axis in [0, 1], "Axis unrecognised")
        if axis == 1:
            self.add_c(dt.matrix_, columns=dt.columns_, dtypes=dt.dtypes_, pos=pos)
        elif axis == 0:
            self.add_r(dt.matrix_, pos=pos)

    def copy(self): 
        """
        :returns: copy of self (maintaining all training prosperties as well, ex batch_size)
        """
        new_ds = DataSet().from_object(self.matrix_, columns=self.columns_, dtypes=self.dtypes_)
        self.train_copy(new_ds)
        return new_ds

    def _get_idx(self, name):
        """
        :param str name: name of column to be searched
        :returns: index of first column named name
        :raises: AssertionError if column named name is not found in the dataset
        """
        assertion(name in self.columns_, "Column not found.")
        return int(np.where(self.columns_ == name)[0][0])

    def _is_categorical(self, idx):
        """
        :param int idx: column index
        :returns: True if column with index idx is categorical, False otherwise
        """
        return not self.dtypes_[idx] in INT_TYPES + FLOAT_TYPES

    # generate new column name if it wasn't provided (util func for add_c)
    def _get_name(self):
        i = 0
        while True:
            if str(i) not in self.columns_:
                return str(i)
            i += 1

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
            data = self.matrix_[key]
            if not isinstance(data, np.ndarray): # if only one row - so one cell - is selected
                return data
            return DataCol.create(data, dtype=self.dtypes_[key[1]]) # return as DataCol
        
        if isinstance(key[1], list | np.ndarray) and isinstance(key[1][0], str): # some rows and columns by names
            key = (key[0], [self._get_idx(k) for k in key[1]])
        return self.matrix_[key]

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
            value = np.array([value], dtype="object")

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

        if value.shape == (1, 1): # if it was a single cell
            self.matrix_[key] = value
        else:
            self.matrix_[key] = value.reshape(self.matrix_[key].shape)

    def __repr__(self):
        self.head()
        return "\r"

    def __str__(self):
        self.head()
        return "\r"

    def __len__(self):
        return len(self.matrix_) if self.matrix_ is not None else 0
