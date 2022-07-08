import numpy as np
import matplotlib.pyplot as plt

from smarty.errors import assertion
from smarty.config import temp_config
from .metrics import mean_squared_error
from .utils import prepare_ds, print_epoch, print_step


LINEAR_SOLVERS = (
    "sgd",
    "norm_eq"
)
"""
:var str sgd: Stohastic Gradient Descent
:var str norm_eq: Normal Equation
"""


class NormalEqSolver:
    def __init__(self, root):
        self.root = root

    def fit(self, ds, *args, **kwargs):
        print_epoch(1, 1)
        print_step(0, 1)

        bias = np.ones((len(ds), 1))
        X = np.c_[bias, ds.get_data_classes()]
        y = ds.get_target_classes()

        x_t = X.T
        all_ = np.linalg.inv(x_t.dot(X)).dot(x_t).dot(y)

        self.root.coefs_ = all_[1:]
        self.root.bias_ = all_[0]

        # turn of verbose for predicting the training performance
        with temp_config(VERBOSE=False):
            y_pred = self.root.predict(ds, *args, **kwargs)

        print_step(1, 1, loss=self.root.loss(ds.get_target_classes(), y_pred))


class SgdSolver:
    def __init__(self, root):
        self.root = root

    def train_predict(self, X_b):
        return X_b.dot(self.root.coefs_) + self.root.bias_

    def gradient_step(self, X_b, y_b):
        y_pred = self.train_predict(X_b)
        const = self.root.learning_rate_ / self.m_
        print(const * np.sum(X_b.T.dot(y_pred - y_b), axis=1))

        self.root.coefs_ -= const * np.sum(X_b.T.dot(y_pred - y_b))
        self.root.bias_ -= const * np.sum(y_pred - y_b)
        return y_pred

    def fit(self, ds, epochs=10, *args, **kwargs):
        self.m_ = len(ds)
        self.root.coefs_ = np.ones((len(ds.data_classes_), 1))
        self.root.bias_ = np.zeros((1, 1))

        src = iter(ds)
        for epoch in range(epochs):
            print_epoch(epoch + 1, epochs)

            for step in range(ds.steps_per_epoch_()):
                X_b, y_b = next(src)
                y_pred = self.gradient_step(X_b, y_b)

                print_step(step + 1, ds.steps_per_epoch_(), loss=self.root.loss(y_b, y_pred))


class LinearRegression:
    """Linear model

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param str solver: solver, one of smarty.models.linear.LINEAR_SOLVERS
    :param float learning_rate: learning rate, used only for solver="sgd"
    :var np.ndarray bias\_: model's bias term
    :var np.ndarray coefs\_: model's coefficients
    :var float learning_rate\_: model's learning rate
    """

    def __init__(self, loss=mean_squared_error, solver="sgd", learning_rate=0.01):
        assertion(solver in LINEAR_SOLVERS, "Solver unrecognised, see models.linear.LINEAR_SOLVERS to see defined one.")
        if solver == "sgd":
            self.solver_ = SgdSolver(self)
        else:
            self.solver_ = NormalEqSolver(self)

        self.loss = loss
        self.bias_ = None
        self.coefs_ = None
        self.learning_rate_ = learning_rate # used only for sgd solver

    def clean_copy(self):
        """
        :returns: new unfited model with same parameters
        """
        return LinearRegression(
            loss=self.loss,
            solver="sgd" if isinstance(self.solver_, SgdSolver) else "norm_eq",
            learning_rate=self.learning_rate_
            )

    @prepare_ds(mode="unsupervised")
    def fit(self, ds, *args, **kwargs):
        """'Trains' the model. If solver="sgd", number of epochs can be set via kwargs (defaults to epochs=10)
        
        :param DataSet ds: a DataSet - data source, needs to have specified target classes
        :returns: self
        """
        self.solver_.fit(ds, *args, **kwargs)
        return self

    @prepare_ds(mode="unsupervised")
    def predict(self, ds, *args, **kwargs):
        """
        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :returns: 2D np.ndarray, where each culumn holds prediction for one of the targets
        :raises: AssertionError if model is not fitted
        """
        assertion(self.bias_ is not None, "Call .fit() first.")

        print_epoch(1, 1, "test")
        y_pred = None
        src = iter(ds)
        for step in range(ds.steps_per_epoch_()):
            x_b = next(src)
            if ds.target_classes_ is not None:
                x_b = x_b[0] # drop target

            y_pred_b = x_b.dot(self.coefs_) + self.bias_
            print_step(step + 1, ds.steps_per_epoch_())

            if y_pred is None:
                y_pred = y_pred_b
            else:
                y_pred = np.r_[y_pred, y_pred_b]
        return y_pred

    @prepare_ds()
    def evaluate(self, ds, loss=None, *args, **kwargs):
        """Evaluates model on the ds according to loss and prints its score
        
        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics. If not provided, loss given on model initialization will be used
        :params args, kwargs: will be passed to .predict()
        """
        y_pred = self.predict(ds, *args, **kwargs)

        if loss is None:
            loss = self.loss
        print(f"Loss: {loss(ds.get_target_classes(), y_pred)}.")

    def plot(self, ds, data_idx=0, target_idx=0, *args, **kwargs):
        """Creates a 2D plot where x-axis is data_idx, and y-axis is target_idx. Plots both their value and prediction curve

        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :param int data_idx: data column index used as x-axis
        :param int target_class: target class index used as y-axis (0 - first target class, 1 - second (if exists) and so on)
        :params args, kwargs: will be passed to .predict()
        """
        y_pred = self.predict(ds, *args, **kwargs)[:, target_idx]
        y = ds.get_target_classes()[:, target_idx]
        x = ds.get_data_classes()[:, data_idx]

        plt.figure(figsize=(12, 8))
        plt.plot(x, y, "b.", alpha=0.3)
        plt.plot(x, y_pred, "r.", alpha=0.5)

        x_min_idx = np.where(x == np.nanmin(x))[0][0]
        x_max_idx = np.where(x == np.nanmax(x))[-1][-1]
        xs = [x[x_min_idx], x[x_max_idx]]
        ys = [y_pred[x_min_idx], y_pred[x_max_idx]]
        plt.plot(xs, ys, "g-", linewidth=4, label="regression line")

        x_lim = [x[x_min_idx], x[x_max_idx]]
        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        y_pred_min = np.nanmin(y_pred)
        y_pred_max = np.nanmax(y_pred)
        y_lim = [y_min if y_min < y_pred_min else y_pred_min, y_max if y_max > y_pred_max else y_pred_max]

        plt.axis([*x_lim, *y_lim])
        plt.legend()
        plt.xlabel(ds.data_classes_[data_idx])
        plt.ylabel(ds.target_classes_[target_idx])
        plt.show()
