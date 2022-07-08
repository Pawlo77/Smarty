# note: dumb models does not support batching

import numpy as np

from smarty.errors import assertion
from .utils import prepare_ds, print_epoch, print_step
from .api import evaluate_model
from .metrics import mean_squared_error


DUMB_SOLVERS = (
    "random",
    "zero"
)
"""
:var str random: model will return random value that it saw during .fit(), for classification only (or regression but it will return only seen targets)
:var str zero: model will return most common value for classification, mean / median value for regression
"""

ZERO_MODES = (
    "regression_mean", 
    "regression_median",
    "classification"
)
"""
:var str regression_mean: smarty.models.dumb.ZeroRuleModel will return mean value for each column
:var str regression_median: smarty.models.dumb.ZeroRuleModel will return meadian value for each column
:var str classification: smarty.models.dumb.ZeroRuleModel will return most common value for each column
"""

# shortcut to evaluate_model that is "dumb", choosen by string parameter mode
def evaluate_dumb(ds, metric=mean_squared_error, solver="random", *args, **kwargs):
    """Shortcut to evaluate_model that is "dumb", choosen by string parameter mode

    :param DataSet ds: a DataSet - data source
    :param function metric: evaluation mertic, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param str solver: one of smarty.models.dumb.DUMB_SOLVERS
    :params args, kwargs: additional params that will be passed to sampling function and model methods
    :raises: AssertionError if solver not recognized
    :returns: score or list of scores

    .. note::
        For solver=zero you need to specify mode, by default it is classification, see smarty.models.dumb.ZERO_MODES
    """
    assertion(solver in DUMB_SOLVERS, "Mode not recognised, see models.dumb.DUMB_SOLVERS to see available modes.")

    if solver =="random":
        model = RandomModel(*args, **kwargs)
    else: # zero
        model = ZeroRuleModel(*args, **kwargs)
    
    return evaluate_model(ds, model, metric, *args, **kwargs)


class RandomModel:
    """For each target column, returns random value from it, classification only"""
    def __init__(self, *args, **kwargs):
        pass

    @prepare_ds()
    def fit(self, ds, *args, **kwargs): 
        """'Trains' the model
        
        :param DataSet ds: a DataSet - data source, needs to have specified target classes
        :returns: self
        """
        self.classes_ = [] # list of unique values for each target_class
        
        print_epoch(1, 1)
        for n, idx in enumerate(ds.target_classes_):
            print_step(n + 1, len(ds.target_classes_))
            self.classes_.append(np.unique(ds[idx]))
        return self
        
    @prepare_ds(mode="unsupervised")
    def predict(self, ds, seed=42, *args, **kwargs):
        """
        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :param int seed: seed for np.random.choice used by model
        :returns: 2D np.ndarray, where each culumn holds prediction for one of the targets
        :raises: AssertionError if model is not fitted
        """
        assertion(self.classes_ != [], "Call .fit() first.")
        np.random.seed(seed)

        predictions = [] # for each target class, random value from it
        print_epoch(1, 1, "test")
        for n, idx in enumerate(self.classes_):
            print_step(n + 1, len(self.classes_))
            predictions.append(np.random.choice(idx, len(ds)).reshape(-1, 1))
        return np.concatenate(predictions, axis=1)

    def clean_copy(self):
        """
        :returns: new unfited model with same parameters
        """
        return RandomModel()


# for each target column, returns most common value from it
class ZeroRuleModel:
    """For each target column it returns predictions according to the rule defined by mode
    
    :param str mode: one of smarty.models.dumb.ZERO_MODES
    """
    def __init__(self, mode="classification", *args, **kwargs):
        assertion(mode in ZERO_MODES, "Wrong mode provided, choose among smarty.models.dumb.ZERO_MODES")
        self.mode_ = mode

    @prepare_ds()
    def fit(self, ds, *args, **kwargs):
        """'Trains' the model
        
        :param DataSet ds: a DataSet - data source, needs to have specified target classes
        :returns: self
        """
        self.classes_ = []

        if self.mode_ == "classification":
            return self._fit_classification(ds, **kwargs)
        return self._fit_regression(ds, **kwargs)

    # returns 2D array, where each culumn holds prediction for one of the targets
    @prepare_ds(mode="unsupervised")
    def predict(self, ds, seed=42, *args, **kwargs):
        """
        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :param int seed: seed for np.random.choice used by model
        :returns: 2D np.ndarray, where each culumn holds prediction for one of the targets
        :raises: AssertionError if model is not fitted
        """
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
        """
        :returns: new unfited model with same parameters
        """
        return ZeroRuleModel(self.mode_)

    # for classification return most common value
    def _fit_classification(self, ds, *args, **kwargs):
        print_epoch(1, 1)
        for n, idx in enumerate(ds.target_classes_):
            print_step(n + 1, len(ds.target_classes_))
            self.classes_.append(max(set(ds[idx]), key=list(ds[idx]).count))
        return self

    # for regression return mean/median
    def _fit_regression(self, ds, *args, **kwargs):
        print_epoch(1, 1)
        for n, idx in enumerate(ds.target_classes_):
            print_step(n + 1, len(ds.target_classes_))
            if self.mode_ == "regression_mean":
                self.classes_.append(np.mean(ds[idx]))
            else: # median
                self.classes_.append(np.median(ds[idx]))
        return self       

