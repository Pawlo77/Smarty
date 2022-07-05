from errors import assertion

import numpy as np


class OneHotEncoder:
    # all functions requires np.ndarray

    def fit(self, cols):
        assertion(isinstance(cols, np.ndarray) and isinstance(cols[0], np.ndarray), "Wrong col type provided")

        self.encoders_ = []
        self.decoders_ = []
        self.classes_ = []
        for i in range(cols.shape[1]): # for each provided column
            self.classes_.append(np.sort(np.unique(cols[:, i])))
            empty_code_ = ["0"] * len(self.classes_[i]) # generate base empty code - string of length - number of classes - filled with "0"

            self.encoders_.append({ # for each class set one of "0"s to "1" - generate one hot
                class_: self._create_code(empty_code_, idx) for idx, class_ in enumerate(self.classes_[i])
            })
            self.decoders_.append({ # reversed encoders_ lookup for faster decoding the codes
                code_: class_ for class_, code_ in self.encoders_[i].items()
            })

        self._num_cols_ = len(self.encoders_)
        return self
    
    def transform(self, cols):
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")

        cols = cols.astype("object") # in case transformed data didn't fit in the original array dtype
        for i in range(self._num_cols_):
            cols[:, i] = [self.encoders_[i][val] for val in cols[:, i]]
        return cols

    def fit_transform(self, cols):
        self.fit(cols)
        return self.transform(cols)

    # translate transformed data back to its original state
    def translate(self, cols):
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")

        cols = cols.astype("object") # in case transformed data didn't fit in the original array
        for i in range(self._num_cols_):
            cols[:, i] = [self.decoders_[i][val] for val in cols[:, i]]
        return cols

    # utility function to help create a lookup by .fit()
    def _create_code(self, empty_code_, idx):
        code_ =  empty_code_[:]
        code_[idx] = "1"
        return "".join(code_)