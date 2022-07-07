import numpy as np

from smart.errors import assertion
from smart.config import temp_config
from .metrics import mean_squared_error
from .utils import prepare_ds, print_epoch, print_step


LINEAR_SOLVERS = (
    "sgd",
    "norm_eq"
)


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
        # print(np.c_[y_pred, y_b, y_pred - y_b])
        const = self.root.learning_rate_ / self.m_

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
        return LinearRegression(
            loss=self.loss,
            solver="sgd" if isinstance(self.solver_, SgdSolver) else "norm_eq",
            learning_rate=self.learning_rate_
            )

    @prepare_ds(mode="unsupervised")
    def fit(self, ds, *args, **kwargs):
        self.solver_.fit(ds, *args, **kwargs)
        return self

    @prepare_ds(mode="unsupervised")
    def predict(self, ds, *args, **kwargs):
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
        y_pred = self.predict(ds, *args, **kwargs)

        if loss is None:
            loss = self.loss
        print(f"Loss: {loss(ds.get_target_classes(), y_pred)}.")

