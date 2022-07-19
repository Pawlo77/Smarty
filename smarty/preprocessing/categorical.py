import numpy as np

from smarty.errors import assertion


class OneHotEncoder:
    """Performs One Hot Encoding on data fed to .fit(). All variables are available after calling .fit() method, which returns OneHotEncoder instance itself.

    :var list of dictionaries encoders\_: list of dictionaries, dict on index i is a map of value -> one hot for a column i
    :var list of dictionaries decoders\_: inversion of encoders\_, maps one hot -> original value
    :var list of str classes\_: list of column names

    .. note::
        .fit() method needs to be called first.
    """ 

    def __init__(self):
        self.encoders_ = []
        self.decoders_ = []
        self.classes_ = []

    def flat_names_(self):
        """
        :returns: 1D array filled with self.classes\_
        :raises: AssertionError if Encoder wasn't fitted 
        """
        assertion(len(self.encoders_) != 0, "Call .fit() first.")
        flat_names = None
        for column in self.classes_:
            if flat_names is None:
                flat_names = column
            else:
                flat_names = np.r_[flat_names, column]
        return flat_names

    def fit(self, cols):
        """Prepares an one-hot map for each column of cols.
        
        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder was already fitted
        """
        assertion(isinstance(cols, np.ndarray) and isinstance(cols[0], np.ndarray), "Wrong col type provided")
        assertion(len(self.encoders_) == 0, "Encoder already fitted.")

        for i in range(cols.shape[1]): # for each provided column
            self.classes_.append(np.sort(np.unique(cols[:, i])))
            empty_code_ = [0] * len(self.classes_[i]) # generate base empty code - string of length - number of classes - filled with "0"

            self.encoders_.append({ # for each class set one of "0"s to "1" - generate one hot
                class_: self._create_code(empty_code_, idx) for idx, class_ in enumerate(self.classes_[i])
            })
            self.decoders_.append({ # reversed encoders_ lookup for faster decoding the codes
                code_: class_ for class_, code_ in self.encoders_[i].items()
            })

        self._num_cols_ = len(self.encoders_)
        return self
    
    def transform(self, cols):
        """Transforms cols into one-hots according to maps created in .fit()

        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder wasn't fitted  or number of columns in cols is different than that in .fit()

        .. danger::
            Function will crash if any value in cols wasn't seen by .fit() method. (wasn't map into encoders\_)
        """
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")
        assertion(len(self.encoders_) != 0, "Call .fit() first.")

        cols = cols.astype("object") # in case transformed data didn't fit in the original array dtype
        res = None
        for i in range(self._num_cols_):
            col = np.array([self.encoders_[i][val] for val in cols[:, i]])
            if res is None:
                res = col
            else:
                res = np.c_[res, col]

        return res

    def fit_transform(self, cols):
        """Fits Encoder with cols and returns their one-hot representation. (calls .fit() than .transform())
        
        :param np.ndarray cols: 2D array, where axis-1 are rows

        .. warning::
            Unlike .fit(), this method will not return OneHotEncoder itself, it should be initialized first if you don't want to loose the maps.
        """
        self.fit(cols)
        return self.transform(cols)

    # translate transformed data back to its original state
    def translate(self, cols):
        """Translates one-hots back to the original value
        
        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder wasn't fitted  or number of columns in cols is different than that returned by .transform()
        
        .. danger::
            Function will crash if any value in cols wasn't seen by .fit() method. (wasn't mapped into decoders\_)
        """
        assertion(cols.shape[1] == len(self.flat_names_()), "Different number of columns provided to that returned by .transform()")
        assertion(len(self.encoders_) != 0, "Call .fit() first.")

        cols = cols.astype("object") # in case transformed data didn't fit in the original array
        res = None
        i = 0
        for idx, unique in enumerate(self.classes_): 
            col = np.array([self.decoders_[idx][tuple(val)] for val in cols[:, i : i + len(unique)]]).reshape(-1, 1)
            if res is None:
                res = col
            else:
                res = np.c_[res, col]
            i += len(unique)
        return res

    # utility function to help create a lookup by .fit()
    def _create_code(self, empty_code_, idx):
        code_ =  empty_code_[:]
        code_[idx] = 1
        return tuple(code_)


class SimpleEncoder:
    """Maps each unique value to unique numerical value

    :var list of dictionaries encoders\_: list of dictionaries, dict on index i is a map of value -> one hot for a column i
    :var list of dictionaries decoders\_: inversion of encoders\_, maps one hot -> original value

    .. note::
        .fit() method needs to be called first.
    """ 

    def __init__(self):
        self.encoders_ = []
        self.decoders_ = []

    def fit(self, cols):
        """Prepares a map for each column of cols.
        
        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder was already fitted
        """
        assertion(isinstance(cols, np.ndarray) and isinstance(cols[0], np.ndarray), "Wrong col type provided")
        assertion(len(self.encoders_) == 0, "Encoder already fitted.")

        for i in range(cols.shape[1]): # for each provided column
            unique = np.sort(np.unique(cols[:, i]))

            self.encoders_.append({ # for each unique value assign it an integer
                u: idx for idx, u in enumerate(unique)
            })
            self.decoders_.append({ # reversed encoders_ lookup for faster decoding the codes
                code_: class_ for class_, code_ in self.encoders_[i].items()
            })

        self._num_cols_ = len(self.encoders_)
        return self
    
    def transform(self, cols):
        """Transforms cols into one-hots according to maps created in .fit()

        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder wasn't fitted  or number of columns in cols is different than that in .fit()

        .. danger::
            Function will crash if any value in cols wasn't seen by .fit() method. (wasn't map into encoders\_)
        """
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")
        assertion(len(self.encoders_) != 0, "Call .fit() first.")

        cols = cols.astype("object") # in case transformed data didn't fit in the original array dtype
        res = None
        for i in range(self._num_cols_):
            col = np.array([self.encoders_[i][val] for val in cols[:, i]])
            if res is None:
                res = col
            else:
                res = np.c_[res, col]

        return res

    def fit_transform(self, cols):
        """Fits Encoder with cols and returns their one-hot representation. (calls .fit() than .transform())
        
        :param np.ndarray cols: 2D array, where axis-1 are rows

        .. warning::
            Unlike .fit(), this method will not return OneHotEncoder itself, it should be initialized first if you don't want to loose the maps.
        """
        self.fit(cols)
        return self.transform(cols)

    # translate transformed data back to its original state
    def translate(self, cols):
        """Translates one-hots back to the original value
        
        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder wasn't fitted  or number of columns in cols is different than that returned by .transform()
        
        .. danger::
            Function will crash if any value in cols wasn't seen by .fit() method. (wasn't mapped into decoders\_)
        """
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to that returned by .transform()")
        assertion(len(self.encoders_) != 0, "Call .fit() first.")

        cols = cols.astype("object") # in case transformed data didn't fit in the original array
        res = None
        for idx in range(self._num_cols_): 
            col = np.array([self.decoders_[idx][val] for val in cols[:, idx]]).reshape(-1, 1)
            if res is None:
                res = col
            else:
                res = np.c_[res, col]
        return res