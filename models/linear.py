import numpy as np

from errors import assertion
from .utils import prepare_ds, print_epoch, print_step


LINEAR_SOLVERS = (
    "sgd",
    "norm_eq"
)


class NormalEqSolver:
    def fit(X, y):
        print_epoch(1, 1)
        print_step(1, 1)
        bias = np.ones((len(X), 1))
        X = np.c_[bias, X]
        x_t = X.T
        all_ = np.linalg.inv(x_t.dot(X)).dot(x_t).dot(y)
        return all_[1:], all_[0]


class LinearRegression:
    def __init__(self, solver="sgd",):
        assertion(solver in LINEAR_SOLVERS, "Solver unrecognised, see models.linear.LINEAR_SOLVERS to see defined one.")
        self.solver_ = solver
        self.bias_ = None
        self.coefs_ = None

    @prepare_ds(mode="unsupervised")
    def fit(self, ds, epochs=10, **kwargs):
        if self.solver_ == "norm_eq":
            self.coefs_, self.bias_ = NormalEqSolver.fit(ds.get_data_classes(), ds.get_target_classes())

        return self

    @prepare_ds(mode="unsupervised")
    def predict(self, ds, **kwargs):
        assertion(self.bias_ is not None, "Call .fit() first.")

        print_epoch(1, 1, "test")
        y_pred = None
        for n, x_b in enumerate(ds):
            if ds.target_classes_ is not None:
                x_b = x_b[0] # drop target

            print_step(n + 1, ds.steps_per_epoch_())
            y_pred_b = x_b.dot(self.coefs_) + self.bias_

            if y_pred is None:
                y_pred = y_pred_b
            else:
                y_pred = np.r_[y_pred, y_pred_b]

        return y_pred


