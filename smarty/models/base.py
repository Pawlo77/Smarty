import numpy as np
import matplotlib.pyplot as plt

from smarty.errors import assertion
from smarty.config import temp_config
from .utils import prepare_ds, print_epoch, print_step, handle_callbacks, print_info


class BaseSolver:
    def __init__(self, root, *args, **kwargs):
        self.root = root

    def set_params(self, params):
        for key, val in params.items(): # root__ endicates that that param belongs to model __dict__ not solver __dict__
            if key.startswith("root__"):
                key = key[6:]
                self.root.__dict__[key] = val
            else:
                self.__dict__[key] = val

    def fit_predict(self, predict, ds, kw):
        if predict:
            # turn of verbose for predicting the training performance
            with temp_config(VERBOSE=False):
                self.root.fitted = True
                y_pred = self.root.predict(ds)
                kw[self.root.loss.__name__] = self.root.loss(ds.get_target_classes(), y_pred)
        return kw

    def get_params(self):
        return {"root_fitted": self.root.fitted}


class MiniBatchGradientDescent(BaseSolver):
    """Mini batch gradient descent base implementation"""

    def __init__(self, *args, **kwargs):
        super(MiniBatchGradientDescent, self).__init__(*args, **kwargs)
        self.root.costs_ = []
        self.root.plot_training = self.plot_training

    def fit(self, ds, epochs=10, *args, **kwargs):
        src = iter(ds)
        for epoch in range(epochs):
            print_epoch(epoch + 1, epochs)

            losses = []
            for step in range(ds.steps_per_epoch_()):
                x_b, y_b = next(src)
                y_pred = self.gradient_step(x_b, y_b)

                loss = self.root.loss(y_b, y_pred)
                losses.append(loss)

                kw = {self.root.loss.__name__: np.mean(losses)}
                print_step(step + 1, ds.steps_per_epoch_(), **kw)
            
            self.root.costs_.append(np.mean(losses))
            
            if not handle_callbacks(self.root, kwargs, losses=losses, epoch=epoch, epochs=epochs):
                return # end training loop

    def plot_training(self):
        """Plots training curves (loss over epochs)"""
        assertion(self.root.fitted, "Call .fit() first.")
        plt.figure(figsize=(8, 6))
        plt.plot(list(range(len(self.root.costs_))), self.root.costs_, "r-")
        plt.xlabel("epoch")
        plt.ylabel(f"{self.root.loss.__name__}")
        plt.title(f"Training {self.root.loss.__name__} over epoch")
        plt.show()

    def predict(self, x_b, *args, **kwargs):
        assertion(False, "Implement predict method.")

    def gradient_step(self, x_b, y_b):
        assertion(False, "Implement gradient_step method.")
    
    def get_params(self):
        kw = super().get_params()
        return kw.update({ # root__ endicates that that param belongs to model __dict__ not solver __dict__
            "root__costs_": self.root.costs_,
            "root__learning_rate_": self.root.learning_rate_,
            "root__loss": self.root.loss,
        })


class BaseModel:
    """Base model, to work it needs to be provided with coefs\_, bias\_, solver\_ and loss function"""
    def __init__(self):
        self.fitted = False

    @prepare_ds()
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
        plt.plot(x, y, "b.", alpha=0.3, label="accual points")
        plt.plot(x, y_pred, "r.", alpha=0.5, label="predicted points")

        x_min_idx = np.where(x == np.nanmin(x))[0][0]
        x_max_idx = np.where(x == np.nanmax(x))[-1][-1]
        xs = [x[x_min_idx], x[x_max_idx]]
        ys = [y_pred[x_min_idx], y_pred[x_max_idx]]
        plt.plot(xs, ys, "g-", linewidth=4, label="decision line")

        x_lim = [x[x_min_idx], x[x_max_idx]]
        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        y_pred_min = np.nanmin(y_pred)
        y_pred_max = np.nanmax(y_pred)
        y_lim = [y_min if y_min < y_pred_min else y_pred_min, y_max if y_max > y_pred_max else y_pred_max]

        plt.axis([*x_lim, *y_lim])
        plt.legend()
        plt.xlabel(ds.columns_[ds.data_classes_[data_idx]])
        plt.ylabel(ds.columns_[ds.target_classes_[target_idx]])
        plt.show()

    @prepare_ds()
    def evaluate(self, ds, loss=None, *args, **kwargs):
        """Evaluates model on the ds according to loss and prints its score
        
        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics. If not provided, loss given on model initialization will be used
        :params args, kwargs: will be passed to .predict()
        :returns: score
        """
        y_pred = self.predict(ds, *args, **kwargs)

        if loss is None:
            loss = self.loss
        score = loss(ds.get_target_classes(), y_pred)
        print_info(f"{loss.__name__}: {score}.")
        return {loss.__name__: score}

    @prepare_ds(mode="prediction")
    def predict(self, ds, *args, **kwargs):
        """
        :param DataSet ds: a DataSet - data source, needs to have specified target classes and shape[1] simmilar to seen in .fit()
        :returns: 2D np.ndarray, where each culumn holds prediction for one of the targets
        :raises: AssertionError if model is not fitted
        """
        assertion(self.fitted, "Call .fit() first.")

        print_epoch(1, 1, "test")
        y_pred = None
        src = iter(ds)
        for step in range(ds.steps_per_epoch_()):
            x_b = next(src)
            if ds.target_classes_ is not None:
                x_b = x_b[0] # drop target

            y_pred_b = self.solver_.predict(x_b, *args, **kwargs)
            print_step(step + 1, ds.steps_per_epoch_())

            if y_pred is None:
                y_pred = y_pred_b
            else:
                y_pred = np.r_[y_pred, y_pred_b]
        return y_pred

    @prepare_ds()
    def fit(self, ds, *args, **kwargs):
        """'Trains' the model.
        
        :param DataSet ds: a DataSet - data source, needs to have specified target classes
        :returns: self
        """
        self.solver_.fit(ds, *args, **kwargs)
        self.fitted = True
        return self

    def get_params(self):
        """Returns dict of parameters allowing exact coping of a model"""
        return self.solver_.get_params()

    def set_params(self, params):
        """Set params to model's and solver's dict
        
        :param dict params: dict of parameters, to indicate that they belong to solver name should start with 'root\_\_'
        """
        self.solver_.set_params(params)