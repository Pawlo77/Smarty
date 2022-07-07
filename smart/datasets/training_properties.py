import numpy as np

from smart.errors import assertion


class TrainingProperty:
    def __init__(self):
        self.batch_size_ = 1
        self.repeat_ = False
        self.shuffle_ = False
        self.drop_reminder_ = True

        # to navigate in _indices_ and to navigate from with data create indices
        self._start_batch_idx_ = self._start_shuffle_idx_ = 0
        self._row_num_ = None # number of rows to be yield

        self._indeces_ = []
        self.target_classes_ = None
        self.data_classes_ = None
        
    # called by native DataSet copy, new - new DataSet object
    def train_copy(self, new):
        new.batch_size_ = self.batch_size_
        new.repeat_ = self.repeat_
        new.shuffle_ = self.shuffle_
        new.drop_reminder_ = self.drop_reminder_
        new.target_classes_ = self.target_classes_
        new.data_classes_ = self.data_classes_

    def steps_per_epoch_(self):
        if self.drop_reminder_:
            return int(len(self) // self.batch_size_)
        return int(np.ceil(len(self) / self.batch_size_))

    # batch dataset, accepts possitive integer value for batch_size
    def batch(self, batch_size=32, drop_reminder=False): 
        assertion(isinstance(batch_size, int) and batch_size >= 1, "Batch size must be an positive integer")
        assertion(isinstance(drop_reminder, bool), "Drop reminder must be a bool")
        self.drop_reminder_ = drop_reminder
        self.batch_size_ = batch_size
        return self # so it can be prepared in one line

    # if opt is an integer - how many times to loop through the dataset, for bool True - infinitive
    def repeat(self, opt=True):
        assertion(isinstance(opt, int) or isinstance(opt, bool), "Repeat must be a bool or integer")
        if isinstance(opt, bool) and opt == True:
            self.repeat_ = np.inf
        else:
            self.repeat_ = opt
        return self

    # integer shuffle size of the dataset, False to disable shuffeling
    def shuffle(self, opt=1000, seed=42):
        assertion(isinstance(opt, int | bool), "Shuffle must be an integer or boolean False")
        assertion(isinstance(seed, int), "Seed must be an integer")
        self.shuffle_ = opt
        self.seed_ = seed
        return self

    # set target classes, accepts key as all other DataSet functions, do not validate it
    def set_target_classes(self, key=None):
        if isinstance(key, str): # single class by name
            key = [self._get_idx(key)]
        elif isinstance(key, int): # single class by idx
            key = [key]
        elif isinstance(key, list | np.ndarray) and isinstance(key[0], str): #multiple class by names
            key = [self._get_idx(k) for k in key]

        self.target_classes_ = key
        self.data_classes_ = [k for k in range(self.get_shape_()[1]) if k not in key] # set rest column as data_classes_

    # returns np.ndarray with columns being one marked as target or none if there is no target
    # it should be called to computations, thus it returns data casted to float
    # assumes all categorical columns are preprocessed to numerical
    def get_target_classes(self):
        dtype = self._get_common_dtype(self.target_classes_)
        if self.target_classes_ is None:
            return np.array([])
        return self.matrix_[:, self.target_classes_].astype(dtype)

    # returns np.ndarray with columns being not marked as target as target or entire matrix_ if no target set
    # it should be called to computations, thus it returns data casted to float
    # assumes all categorical columns are preprocessed to numerical
    def get_data_classes(self):
        dtype = self._get_common_dtype(self.data_classes_)
        if self.target_classes_ is None:
            return self.matrix_.astype(dtype)
        return self.matrix_[:, self.data_classes_].astype(dtype)

    def _get_common_dtype(self, cols):
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
                idxs = self._indeces_[self._start_batch_idx_:] # use remaining one
                self._generate_indeces()
                self._start_batch_idx_ = size - len(idxs) # how many we will take from next _indeces_
                idxs = np.r_[idxs, self._indeces_[:self._start_batch_idx_]] # fill rest with new toss
            else:
                idxs = self._indeces_[self._start_batch_idx_:self._start_batch_idx_ + size] # fill all from the current toss
                self._start_batch_idx_ += size # mark to start next iteration by size later

            self._row_num_ -= size # we used next size rows
            idxs = idxs.reshape(-1, 1)
            if self.target_classes_ is not None: # supervised learning
                return self.matrix_[idxs, self.data_classes_], self.matrix_[idxs, self.target_classes_]
            else: # unsupervised learning or not learning at all
                return self.matrix_[idxs, :]
        else: # we yielded everything we could
            raise StopIteration

    def __iter__(self):
        self._row_num_ = len(self)
        self._start_batch_idx_ = self._start_shuffle_idx_ = 0
        self._generate_indeces()
        return self

