import numpy as np

from smarty.errors import assertion


class NormalizeSolver:
    def fit(self, cols):
        # calculate min and max for each given column
        self.mins_ = np.nanmin(cols, axis=0)
        self.maxs_ = np.nanmax(cols, axis=0)
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
        self.stds_ = np.nanstd(cols, axis=0)
        self.means_ = np.nanmean(cols, axis=0)
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
    """Performs basic numerical data preprocessing according to choosen stategy. All variables are available after calling .fit() method, which returns StandardScaler instance itself.

    :param str strategy: "normalize" for data normalization or "standarize" for data standarization
    :var list strategies\_: List of avalibable strategies
    :var int solver.\_num_cols\_: Number of columns provided in .fit() method
    :var np.ndarray solver.stds\_: List containing each column standard deviation, available only with **strategy="standarize"**
    :var np.ndarray solver.means\_: List containing each column mean, available only with **strategy="standarize"**
    :var np.ndarray solver.mins\_: List containing each column minimum value, available only with **strategy="normalize"**
    :var np.ndarray solver.maxs\_: List containing each column maximum value, available only with **strategy="normalize"**


    .. note::
        .fit() method needs to be called first.
    """

    def __init__(self, strategy="standarize"):
        self.strategies_ = ["normalize", "standarize"]
        assertion(strategy in self.strategies_, "Strategy not recognized. To see avalibable, call StandardScaler().strategies_")
        self.strategy_ = strategy
        self._solver_ = None

    def fit(self, cols):
        """Calculates neccesery statistical data to perform later transformations.
        
        :param cols: 2D array, where axis-1 are rows
        :type cols: np.ndarray
        :raises: AssertionError if Encoder was already fitted
        """
        assertion(isinstance(cols, np.ndarray) and isinstance(cols[0], np.ndarray), "Wrong col type provided")
        assertion(self._solver_ is None, "Scaler already fitted.")
        cols = self.cast(cols) # make sure they are float

        if self.strategy_ == "standarize":
            self._solver_ = StandarizeSolver().fit(cols)
        elif self.strategy_ == "normalize":
            self._solver_ = NormalizeSolver().fit(cols)

        return self
    
    def transform(self, cols):
        """Transforms cols according to choosen strategy.

        :param cols: 2D array, where axis-1 are rows
        :type cols: np.ndarray
        :raises: AssertionError if Encoder wasn't fitted  or number of columns in cols is different than that in .fit()

        .. danger::
            Function will crash if any value in cols wasn't seen by .fit() method. (wasn't map into encoders\_)
        """
        assertion(self.strategy_ is not None, "Call .fit() first")
        cols = self._solver_.cast(cols)
        return self._solver_.transform(cols)

    def fit_transform(self, cols):
        """Fits Scaler with cols and returns their one-hot representation. (calls .fit() than .transform())
        
        :param np.ndarray cols: 2D array, where axis-1 are rows

        .. warning::
            Unlike .fit(), this method will not return StandardScaler itself, it should be initialized first if you want to save the stats for later use.
        """
        self.fit(cols)
        return self._solver_.transform(cols)

    # translate transformed data back to its original state
    def translate(self, cols):
        """Translates preprocessed data as close as possible to their original values
        
        :param np.ndarray cols: 2D array, where axis-1 are rows
        :raises: AssertionError if Encoder wasn't fitted  or number of columns in cols is different than that in .fit()
        """
        assertion(self.strategy_ is not None, "Call .fit() first")
        cols = self._solver_.cast(cols) # make sure they are float
        return self._solver_.translate(cols)

    # to compute statictical data transform each column to float
    def cast(self, cols):
        return cols.astype("f")


