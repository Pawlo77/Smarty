# note: dumb models does not support batching

import numpy as np

from smart.errors import assertion
from .utils import prepare_ds, print_epoch, print_step
from .api import evaluate_model
from .metrics import mean_squared_error


DUMB_SOLVERS = (
    "random",
    "zero"
)

# shortcut to evaluate_model that is "dumb", choosen by string parameter mode
def evaluate_dumb(ds, metric=mean_squared_error, solver="random", *args, **kwargs):
    assertion(solver in DUMB_SOLVERS, "Mode not recognised, see models.dumb.DUMB_SOLVERS to see available modes.")

    if solver =="random":
        model = RandomModel(*args, **kwargs)
    else: # zero
        model = ZeroRuleModel(*args, **kwargs)
    
    return evaluate_model(ds, model, metric, *args, **kwargs)


# for each target column, returns random value from it
# classification only
class RandomModel:
    def __init__(self, *args, **kwargs):
        pass

    @prepare_ds()
    def fit(self, ds, *args, **kwargs): 
        self.classes_ = [] # list of unique values for each target_class
        
        print_epoch(1, 1)
        for n, idx in enumerate(ds.target_classes_):
            print_step(n + 1, len(ds.target_classes_))
            self.classes_.append(np.unique(ds[idx]))
        return self
        
    # returns 2D array, where each culumn holds prediction for one of the targets
    @prepare_ds(mode="unsupervised")
    def predict(self, ds, seed=42, *args, **kwargs):
        assertion(self.classes_ != [], "Call .fit() first.")
        np.random.seed(seed)

        predictions = [] # for each target class, random value from it
        print_epoch(1, 1, "test")
        for n, idx in enumerate(self.classes_):
            print_step(n + 1, len(self.classes_))
            predictions.append(np.random.choice(idx, len(ds)).reshape(-1, 1))
        return np.concatenate(predictions, axis=1)

    # returns new unfited model with same parameters
    def clean_copy(self):
        return RandomModel()


# for each target column, returns most common value from it
class ZeroRuleModel:
    def __init__(self, mode="classification", **kwargs):
        self.modes_ = ["classification", "regression_mean", "regression_avg"]
        assertion(mode in self.modes_, "Wrong mode provided, choose among self.modes_")
        self.mode_ = mode

    @prepare_ds()
    def fit(self, ds, *args, **kwargs):
        self.classes_ = []

        if self.mode_ == "classification":
            return self._fit_classification(ds, **kwargs)
        return self._fit_regression(ds, **kwargs)

    # returns 2D array, where each culumn holds prediction for one of the targets
    @prepare_ds(mode="unsupervised")
    def predict(self, ds, seed=42, *args, **kwargs):
        assertion(self.classes_ != [], "Call .fit() first.")
        np.random.seed(seed)

        print_epoch(1, 1, "test")
        predictions = [] # for each target class, our "predicted" value
        for n, c in enumerate(self.classes_):
            print_step(n + 1, len(self.classes_))
            predictions.append(np.full(len(ds), c).reshape(-1, 1))
        return np.concatenate(predictions, axis=1)

    # returns new unfited model with same parameters
    def clean_copy(self):
        return ZeroRuleModel(self.mode_)

    # for classification return most common value
    def _fit_classification(self, ds, *args, **kwargs):
        print_epoch(1, 1)
        for n, idx in enumerate(ds.target_classes_):
            print_step(n + 1, len(ds.target_classes_))
            self.classes_.append(max(set(ds[idx]), key=list(ds[idx]).count))
        return self

    # for regression return mean/avg
    def _fit_regression(self, ds, *args, **kwargs):
        print_epoch(1, 1)
        for n, idx in enumerate(ds.target_classes_):
            print_step(n + 1, len(ds.target_classes_))
            if self.mode_ == "regression_mean":
                self.classes_.append(np.mean(ds[idx]))
            else: # avg
                self.classes_.append(np.average(ds[idx]))
        return self       

