import numpy as np

from smarty.errors import assertion
from smarty.metrics import accuracy
from .utils import prepare_ds, print_epoch, print_step
from .base import BaseModel, BaseSolver


class NaiveSolver(BaseSolver):
    def fit(self, ds, predict=True, *args, **kwargs):
        print_epoch(1, 1)
        print_step(0, 1)
        self.split_by_class(ds)

        kw = {}
        self.fit_predict(predict, ds, kw)
        print_step(1, 1, **kw)

    @prepare_ds(batch_size=1)
    def split_by_class(self, ds, *args, **kwargs):
        assertion(len(ds.target_classes_) == 1, "Suport only with one target class.") 
        target = np.unique(ds.get_target_classes())
        self.root.map_ = {val: idx for idx, val in enumerate(target)}

        self.root.stats_ = [[] for _ in range(len(target))]
        for x_b, y_b in ds:
            idx = self.root.map_[y_b[0, 0]]
            self.root.stats_[idx].append(x_b)

        for i in range(len(target)):
            stats = np.array(self.root.stats_[i])
            # mean for each column, std for each column, proba of the class
            self.root.stats_[i] = (np.nanmean(stats, axis=0), np.nanstd(stats, axis=0), len(stats) / len(ds))

    def calculate_proba(self, x_b, mean, stdev):
        exponent = np.exp(-((x_b - mean) ** 2 / (2 * stdev ** 2)))
        return np.prod((1 / np.sqrt(2 * np.pi) * stdev) * exponent, axis=1)

    def predict_proba(self, x_b):
        proba = None
        for mean, std, pr in self.root.stats_:
            p = pr * self.calculate_proba(x_b, mean, std)
            if proba is None:
                proba = np.array(p)
            else:
                proba = np.c_[proba, p]
        return proba

    def predict(self, x_b, return_proba=False, *args, **kwargs):
        proba = self.predict_proba(x_b)
        
        if return_proba:
            return proba

        y_pred = np.where(proba == np.max(proba, axis=1).reshape(-1, 1))[1]
        y_pred = np.array([self.root.map_[idx] for idx in y_pred]).reshape(-1, 1)
        return y_pred

    def get_params(self):
        kw = super().get_params()
        
        return kw.update({
            "root__map_": self.root.map_, 
            "root__stats_": self.root.stats_,
            "root__loss": self.root.loss
        })


class NaiveBayesCLassifier(BaseModel):
    """NaiveBayesCLassifier model

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    """
    def __init__(self, loss=accuracy, *args, **kwargs):
        super(NaiveBayesCLassifier, self).__init__(*args, **kwargs)
        self.solver_ = NaiveSolver(self)
        self.loss = loss

    def clean_copy(self):
        return NaiveBayesCLassifier(self.loss)
