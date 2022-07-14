import numpy as np

from smarty.errors import assertion
from smarty.config import temp_config
from .metrics import mean_squared_error, accuracy
from .utils import prepare_ds, print_epoch, print_step
from .base import MiniBatchGradientDescent, BaseModel


LINEAR_SOLVERS = (
    "mbgd",
    "norm_eq"
)
"""
:var str mbgd: Mini-Batch Gradient Descent
:var str norm_eq: Normal Equation
"""


class NormalEqSolver:
    def __init__(self, root):
        self.root = root
        self.root.bias_ = None
        self.root.coefs_ = None

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


class LinearSgdSolver(MiniBatchGradientDescent):
    def __init__(self, *args, **kwargs):
        super(LinearSgdSolver, self).__init__(*args, **kwargs)


class LogisticSgdSolver(MiniBatchGradientDescent):
    def __init__(self, *args, **kwargs):
        super(LogisticSgdSolver, self).__init__(*args, **kwargs)

    def predict(self, X_b):
        y_pred = super().predict(X_b)
        return (1.0 / (1.0 + np.exp(-y_pred))).astype("i")


class PerceptronSolver(MiniBatchGradientDescent):
    def __init__(self, *args, **kwargs):
        super(PerceptronSolver, self).__init__(*args, **kwargs)

    def predict(self, X_b):
        y_pred = super().predict(X_b)
        idxs = np.where(y_pred > self.root.threshold_)
        y_pred = np.zeros(y_pred.shape)
        y_pred[idxs] = 1
        return y_pred


class LinearRegression(BaseModel):
    """Linear model

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param str solver: solver, one of smarty.models.linear.LINEAR_SOLVERS
    :param float learning_rate: learning rate, used only for solver="sgd"
    :var np.ndarray bias\_: model's bias term
    :var np.ndarray coefs\_: model's coefficients
    :var float learning_rate\_: model's learning rate

    .. note::
        If you are using solver="sgd", you can plot training curve via .plot_training() or see each eopch losses - list at .solver_.costs_
    """

    def __init__(self, loss=mean_squared_error, solver="mbgd", learning_rate=0.0001):
        assertion(solver in LINEAR_SOLVERS, "Solver unrecognised, see models.linear.LINEAR_SOLVERS to see defined one.")
        if solver == "mbgd":
            self.solver_ = LinearSgdSolver(self)
        else:
            self.solver_ = NormalEqSolver(self)

        self.loss = loss
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


class LogisticRegression(BaseModel):
    """Logistic binary classifier

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param float learning_rate: learning rate
    :var np.ndarray bias\_: model's bias term
    :var np.ndarray coefs\_: model's coefficients
    :var float learning_rate\_: model's learning rate
    """

    def __init__(self, loss=accuracy, learning_rate=0.0001):
        self.loss = loss
        self.solver_ = LogisticSgdSolver(self)
        self.learning_rate_ = learning_rate

    def fit(self, ds, *args, **kwargs):
        # make sure the target class is correct for binary classification
        target = ds.get_target_classes()
        assertion(target.shape[1] == 1, "Binary classifier can have only one class")
        unique = np.unique(target)
        assertion(list(unique) == [0, 1], "Target class must consist only of 0 and 1")

        return super().fit(ds, *args, **kwargs)

    def clean_copy(self):
        """
        :returns: new unfited model with same parameters
        """
        return LogisticRegression(
            loss=self.loss,
            learning_rate=self.learning_rate_
            )


class Perceptron(BaseModel):
    """Perceptron binary classifier

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param float learning_rate: learning rate
    :param float threshold: values higher than threshold will be classified as 1, rest as 0
    :var np.ndarray bias\_: model's bias term
    :var np.ndarray coefs\_: model's coefficients
    :var float learning_rate\_: model's learning rate
    """

    def __init__(self, loss=accuracy, learning_rate=0.0001, threshold=0.1):
        self.loss = loss
        self.solver_ = PerceptronSolver(self)
        self.learning_rate_ = learning_rate
        self.threshold_ = threshold

    def fit(self, ds, *args, **kwargs):
        # make sure the target class is correct for binary classification
        target = ds.get_target_classes()
        assertion(target.shape[1] == 1, "Binary classifier can have only one class")
        unique = np.unique(target)
        assertion(list(unique) == [0, 1], "Target class must consist only of 0 and 1")

        return super().fit(ds, *args, **kwargs)

    def clean_copy(self):
        """
        :returns: new unfited model with same parameters
        """
        return Perceptron(
            loss=self.loss,
            learning_rate=self.learning_rate_,
            threshold=self.threshold_
            )

    def seek_threshold(self, ds, min_t=0.0, max_t=1.0, bins=5, mode="max", *args, **kwargs):
        """Trains itself and keeps coefs and bias with highest accuracy
        
        :param float min_t: minimum threshold value
        :param float max_t: maximum threshold value
        :param int bins: number of bins to check
        :param str mode: 'max' - seeks for model with highest loss, 'min' - seeks for model with lowest loss
        :returns: best found score (measurred via self.evaluate)
        """
        assertion(mode in ['max', 'min'], "Mode not recognized, allowed are 'min' or 'max'")

        best_score = best_coefs = best_bias = best_threshold = None
        for threshold in np.linspace(min_t, max_t, bins):
            self.threshold_ = threshold
            self.solver_ = PerceptronSolver(self)
            
            self.fit(ds, *args, **kwargs)
            score = self.evaluate(ds, *args, **kwargs)

            if best_score is None or (mode == "max" and score > best_score) or (mode == "min" and score < best_score):
                best_threshold = threshold
                best_score = score
                best_coefs = self.coefs_
                best_bias = self.bias_

        self.coefs_ = best_coefs
        self.bias_ = best_bias
        self.threshold_ = best_threshold
        return best_score
