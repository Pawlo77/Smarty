import numpy as np

from smarty.metrics import accuracy, root_mean_squared_error
from smarty.errors import assertion
from smarty.models.utils import print_epoch, print_step
from smarty.models.base import BaseModel, BaseSolver


class KNNSolver(BaseSolver):
    def euclidean_distance(self, x1, x2): 
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def get_neighbors(self, x_b, row, k=5, **kwargs):
        # return x_b's indexes of k nearest neighbors for row
        distances = self.euclidean_distance(x_b, row)
        distances = list(enumerate(distances))
        distances.sort(key = lambda x: x[1])
        distances = distances[1: min(len(distances), k + 1)]
        return np.array(distances)[:, 0].astype(np.int64)

    def predict_classification(self, x_b, *args, **kwargs):
        y_pred = []
        for row in x_b:
            idxs = self.get_neighbors(self.root.ds.get_data_classes(), row, **kwargs)
            targets = self.root.ds.get_target_classes()[idxs, :]

            value, counts = np.unique(targets, return_counts=True)
            y_pred.append(value[np.argmax(counts)])

        return np.array(y_pred).reshape(-1, 1)

    def predict_regression(self, x_b, *args, **kwargs):
        y_pred = []
        for row in x_b:
            idxs = self.get_neighbors(self.root.ds.get_data_classes(), row, **kwargs)
            targets = self.root.ds.get_target_classes()[idxs, :]

            if self.root.strategy == "mean":
                y_pred.append(np.mean(targets))
            else:
                y_pred.append(np.median(targets))
        
        return np.array(y_pred).reshape(-1, 1)

    def fit(self, ds, predict=True, *args, **kwargs):
        print_epoch(1, 1)
        self.root.ds = ds
        print_step(0, 1)

        kw = {}
        self.fit_predict(predict, ds, kw)
        print_step(1, 1, **kw)
    
    def predict(self, x_b, *args, **kwargs):
        if self.root.mode_ == "classification":
            return self.predict_classification(x_b, *args, **kwargs)
        return self.predict_regression(x_b, *args, **kwargs)

    def get_params(self):
        kw = super().get_params()
        
        return kw.update({
            "root__loss": self.root.loss,
            "root__ds": self.root.ds,
        })


class KNNClassifier(BaseModel):
    """K Nearest Neighbors Classifier
    Default k is 5, to change it pass it in .predict(), .fit() or .evaluate()

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    """
    def __init__(self, loss=accuracy, *args, **kwargs):
        super(KNNClassifier, self).__init__(*args, **kwargs)
        self.mode_ = "classification"
        self.loss = loss
        self.solver_ = KNNSolver(self)

    def clean_copy(self):
        return KNNClassifier(loss=self.loss)


class KNNRegressor(BaseModel):
    """K Nearest Neighbors Regressor
    Default k is 5, to change it pass it in .predict(), .fit() or .evaluate()

    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    """
    def __init__(self, loss=root_mean_squared_error, strategy="mean", *args, **kwargs):
        super(KNNRegressor, self).__init__(*args, **kwargs)
        self.strategies = ["mean", "median"]
        assertion(strategy in self.strategies, f"Strategy not recognized, choose from: {' '.join(self.strategies)}")
    
        self.mode_ = "regression"
        self.strategy = strategy
        self.loss = loss
        self.solver_ = KNNSolver(self)

    def clean_copy(self):
        return KNNRegressor(loss=self.loss)