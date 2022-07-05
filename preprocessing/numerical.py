from errors import assertion

import numpy as np


class NormalizeSolver:
    def fit(self, cols):
        # calculate min and max for each given column
        self.mins_ = np.min(cols, axis=0)
        self.maxs_ = np.max(cols, axis=0)
        self._num_cols_ = len(self.mins_)
        return self

    def transform(self, cols):
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")

        for i in range(self._num_cols_): # normalize each column
            cols[:, i] = (cols[:, i] - self.mins_[i]) / (self.maxs_[i] - self.mins_[i])
        return cols

    def translate(self, cols):
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")

        for i in range(self._num_cols_): # reverse normalization
            cols[:, i] = cols[:, i] * (self.maxs_[i] - self.mins_[i]) + self.mins_[i]
        return cols
        

class StandarizeSolver:
    def fit(self, cols):
        # calculate std and mean for each given column
        self.stds_ = np.std(cols, axis=0)
        self.means_ = np.mean(cols, axis=0)
        self._num_cols_ = len(self.stds_)
        return self

    def transform(self, cols):
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")

        for i in range(self._num_cols_): # standarize each column
            cols[:, i] = (cols[:, i] - self.means_[i]) / self.stds_[i]
        return cols

    def translate(self, cols):
        assertion(cols.shape[1] == self._num_cols_, "Different number of columns provided to one in .fit()")

        for i in range(self._num_cols_): # reversed standarization
            cols[:, i] = cols[:, i] * self.stds_[i] + self.means_[i]
        return cols


class StandardScaler:
    # all functions requires np.ndarray
    def __init__(self, strategy="standarize"):
        self.strategies_ = ["normalize", "standarize"]
        assertion(strategy in self.strategies_, "Strategy not recognized. To see avalibable, call StandardScaler().strategies_")
        self.strategy_ = strategy
        self.solver_ = None

    def fit(self, cols):
        assertion(isinstance(cols, np.ndarray) and isinstance(cols[0], np.ndarray), "Wrong col type provided")
        cols = self.cast(cols) # make sure they are float

        if self.strategy_ == "standarize":
            self.solver = StandarizeSolver().fit(cols)
        elif self.strategy_ == "normalize":
            self.solver = NormalizeSolver().fit(cols)

        return self
    
    def transform(self, cols):
        assertion(self.strategy_ is not None, "Call .fit() first")
        cols = self.solver.cast(cols)
        return self.solver.transform(cols)

    def fit_transform(self, cols):
        self.fit(cols)
        return self.solver.transform(cols)

    # translate transformed data back to its original state
    def translate(self, cols):
        assertion(self.strategy_ is not None, "Call .fit() first")
        cols = self.solver.cast(cols) # make sure they are float
        return self.solver.translate(cols)

    # to compute statictical data transform each column to float
    def cast(self, cols):
        return cols.astype("f")


