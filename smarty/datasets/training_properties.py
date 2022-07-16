import numpy as np

from smarty.errors import assertion


class TrainingProperty:
    """Training methods for a DataSet
    
    :var int batch_size\_: batch size
    :var int | bool | np.inf repeat\_: how many times to repeat the dataset (False - do not repeat, np.inf will repeat forever)
    :var bool | int shuffle\_: if False yielded batches will look the same, otherwise will be shuffled in buckets of size shuffle\_
    :var bool drop_reminder\_: if True last batch of each full dataset iteration is dropped if it is not full
    :var list target_classes\_: target classes that are yielded separately from data_classes\_ during training (for supervised algorithms)
    :var list data_classes\_: data classes, if not set entire dataset is presumed to be consisted only from data_classes\_

    .. note::
        It's recommended to set upper variables via proper methods (look below)
    """

    def __init__(self):
        self.batch_size_ = 1
        self.repeat_ = False
        self.shuffle_ = False
        self.drop_reminder_ = True
        self.seed_ = 42

        # to navigate in _indices_ and to navigate from with data create indices
        self._start_batch_idx_ = self._start_shuffle_idx_ = 0
        self._row_num_ = None # number of rows to be yield

        self._indeces_ = []
        self.target_classes_ = None
        self.data_classes_ = None
        
    # called by native DataSet copy, new - new DataSet object
    def train_copy(self, new):
        """Sets new DataSet instance's traiing variables equal to its training variables
        
        :param DataSet new: new DataSet instance
        """
        new.batch_size_ = self.batch_size_
        new.repeat_ = self.repeat_
        new.shuffle_ = self.shuffle_
        new.seed_ = self.seed_
        new.drop_reminder_ = self.drop_reminder_
        new.target_classes_ = self.target_classes_
        new.data_classes_ = self.data_classes_

    def steps_per_epoch_(self):
        """Number of steps per epoch - number of batches yileded at each compleate dataset iteration."""
        if self.drop_reminder_:
            return int(len(self) // self.batch_size_)
        return int(np.ceil(len(self) / self.batch_size_))

    def batch(self, batch_size=32, drop_reminder=False): 
        """Batches dataset
        
        :param positive int batch_size: batch size
        :param bool drop_reminder: weather or not to drop reminder
        :returns: self
        """
        assertion(isinstance(batch_size, int) and batch_size >= 1, "Batch size must be an positive integer")
        assertion(isinstance(drop_reminder, bool), "Drop reminder must be a bool")
        self.drop_reminder_ = drop_reminder
        self.batch_size_ = batch_size
        return self # so it can be prepared in one line

    def repeat(self, opt=True):
        """Repeats the dataset

        :param bool | int opt: False - do not repeat (1 iteration), True - infinitive, int - how many times to repeat
        """
        assertion(isinstance(opt, int) or isinstance(opt, bool), "Repeat must be a bool or integer")
        if isinstance(opt, bool) and opt == True:
            self.repeat_ = np.inf
        else:
            self.repeat_ = opt
        return self

    def shuffle(self, opt=1000, seed=42):
        """Shuffles the datasets 

        :param bool | int opt: False - do not shuffle, int - shuffle box size
        :param int seed: random seed for numpy
        """
        assertion(isinstance(opt, int | bool), "Shuffle must be an integer or boolean False")
        assertion(isinstance(seed, int), "Seed must be an integer")
        self.shuffle_ = opt
        self.seed_ = seed
        return self

    def set_target_classes(self, key=None):
        """Allows to set the target classes for the dataset (supervised learning)

        :param int | list | str key: column name (str), list of column names (list of strs), column index (int) or list of columns indexes (list of ints)
        """
        if isinstance(key, str): # single class by name
            key = [self._get_idx(key)]
        elif isinstance(key, int): # single class by idx
            key = [key]
        elif isinstance(key, list | np.ndarray) and isinstance(key[0], str): #multiple class by names
            key = [self._get_idx(k) for k in key]

        for idx in range(len(key)):
            if key[idx] < 0:
                key[idx] = self.get_shape_()[1] + key[idx]

        self.target_classes_ = key
        self.data_classes_ = [k for k in range(self.get_shape_()[1]) if k not in key] # set rest column as data_classes_
        return self

    # _rows - specified rows indexes to be returned, only for internal use
    def get_target_classes(self, _rows=None):
        """
        :returns: np.ndarray of columns being marked as target or an empty np.ndarray if there is no target

        .. note:: 
            It presumes all categorical columns are preprocessed to numerical one, and try to cast output array to float to allow further computations
        """
        dtype = self._get_common_dtype(self.target_classes_)
        if self.target_classes_ is None:
            return np.array([])

        if _rows is not None:
            return self.matrix_[_rows, self.target_classes_].astype(dtype)
        return self.matrix_[:, self.target_classes_].astype(dtype)

    # _rows - specified rows indexes to be returned, only for internal use
    def get_data_classes(self, _rows=None):
        """
        :returns: np.ndarray of columns being marked as data or entire matrix\_ if no target column is selected

        .. note:: 
            It presumes all categorical columns are preprocessed to numerical one, and try to cast output array to float to allow further computations
        """
        dtype = self._get_common_dtype(self.data_classes_)
        if self.target_classes_ is None:
            if _rows is not None:
                return self.matrix_[_rows, :].astype(dtype)
            return self.matrix_.astype(dtype)

        if _rows is not None:
            return self.matrix_[_rows, self.data_classes_].astype(dtype)
        return self.matrix_[:, self.data_classes_].astype(dtype)

    def _get_common_dtype(self, cols):
        """Searches common dtype of selected data columns cols, it works under assumption that each of these columns is alredy numerical

        :param int | list | str key: column name (str), list of column names (list of strs), column index (int) or list of columns indexes (list of ints)
        :raturns: np.float32 or np.float64
        """
        dtypes = self.dtypes_[cols]
        largest = sorted(list(dtypes), key = lambda dt: dt.itemsize)[-1]
        if largest.itemsize <= 4:
            return np.float32
        else:
            return np.float64

    def _generate_indeces(self):
        # recurent function to generate list of indeces of size indeces_size,
        # starting with start_idx, where no index exceeds max_idx, and next 
        # overlaps (if they are neccesery) starts from 0
        def generate_helper(start_idx, max_idx, indeces_size):
            if indeces_size == 0:
                return
            
            step_size = min(max_idx - start_idx, indeces_size)
            step_indeces = np.arange(start_idx, start_idx + step_size)

            next_step_start_idx = start_idx + step_size if start_idx + step_size < max_idx else 0
            next_step_indeces = generate_helper(next_step_start_idx, max_idx, indeces_size - step_size)

            if next_step_indeces is not None:
                return np.r_[step_indeces, next_step_indeces]
            return step_indeces

        max_idx_ = len(self) # number of rows in dataset
        indeces_size = self.shuffle_ if self.shuffle_ is not False else 1000 # how large indeces we want to be
        self._indeces_ = generate_helper(self._start_shuffle_idx_, max_idx_, indeces_size)
        
        self._start_shuffle_idx_ = self._indeces_[-1] + 1
        if self._start_shuffle_idx_ == max_idx_: # if out of row range - last generated idx is max_idx
            self._start_shuffle_idx_ = 0 # mark to start nex generating from 0

        if self.shuffle_ is not False:
            np.random.seed(self.seed_)
            np.random.shuffle(self._indeces_)

    def __next__(self):
        def next_epoch():
            if self.repeat_ is not False and self._row_num_ <= 0: # repeat dataset next time
                self._row_num_ = len(self)
                self.repeat_ -= 1 

                if self.repeat_ == 0:
                    raise StopIteration

        next_epoch()
        if self._row_num_ > 0: # if we still have some data to yield
            size = min(self._row_num_, self.batch_size_)

            if size != self.batch_size_ and self.drop_reminder_:
                next_epoch()

            # if sice exceedes remaining _incedes_
            if self._start_batch_idx_ + size > len(self._indeces_):
                while self._start_batch_idx_ + size > len(self._indeces_):
                    idxs = self._indeces_[self._start_batch_idx_:] # use remaining one
                    self._generate_indeces()
                    self._start_batch_idx_ = size - len(idxs) # how many we will take from next _indeces_
                    idxs = np.r_[idxs, self._indeces_[:self._start_batch_idx_]] # fill rest with new toss
            else:
                idxs = self._indeces_[self._start_batch_idx_:self._start_batch_idx_ + size] # fill all from the current toss
                self._start_batch_idx_ += size # mark to start next iteration by size later

            self._row_num_ -= size # we used next size rows
            idxs = idxs.reshape(-1, 1)
            if self.target_classes_ is not None and len(self.target_classes_) != 0: # supervised learning
                return self.get_data_classes(idxs), self.get_target_classes(idxs)
            else: # unsupervised learning or not learning at all
                return self.get_data_classes(idxs)
        else: # we yielded everything we could
            raise StopIteration

    def __iter__(self, train=False):
        self._row_num_ = len(self)
        self._start_batch_idx_ = self._start_shuffle_idx_ = 0
        self._generate_indeces()
        return self

