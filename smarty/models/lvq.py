import numpy as np

from smarty.metrics import accuracy
from smarty.models.base import BaseModel, MiniBatchGradientDescent


class LVQSolver(MiniBatchGradientDescent):
    def initialize_codebooks(self, ds):
        idxs = np.arange(len(ds))
        if self.root.n_codebooks_ is None:
            self.root.n_codebooks_ = max(len(ds) // 20, 3) # if user didn't set n_codebooks_, use auto
                
        def helper(max_idx, src):
            res = None
            for i in range(max_idx):
                np.random.shuffle(idxs)
                idx = idxs[:self.root.n_codebooks_]

                if res is None:
                    res = src[idx, i]
                else:
                    res = np.c_[res, src[idx, i]]

            if len(res.shape) == 1:
                res = res[..., np.newaxis]
            return res

        return helper(len(ds.data_classes_), ds.get_data_classes()), helper(len(ds.target_classes_), ds.get_target_classes())

    def euclidean_distance(self, x1, x2): 
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def get_best_unit(self, codebooks, row, **kwargs):
        distances = self.euclidean_distance(codebooks, row) # get unit from codebooks that is the closest to row
        return np.argmin(distances) # return its index

    def fit(self, ds, *args, **kwargs):
        self.root.m_ = len(ds)
        self.root.codebooks_, self.root.codebooks_targets_ = self.initialize_codebooks(ds)
        return super().fit(ds, *args, **kwargs)
    
    def gradient_step(self, x_b, y_b):
        idxs = [self.get_best_unit(self.root.codebooks_, row) for row in x_b]
        const = self.root.learning_rate_ #/ self.root.m_
        error = x_b - self.root.codebooks_[idxs, :]

        for i in range(len(x_b)):
            if np.all(y_b[i] == self.root.codebooks_targets_[idxs[i]]):
                self.root.codebooks_[idxs[i], :] += const * error[i, :]
            else:
                self.root.codebooks_[idxs[i], :] -= const * error[i, :]

        return self.root.codebooks_targets_[idxs, :]

    def predict(self, x_b):
        idxs = [self.get_best_unit(self.root.codebooks_, row) for row in x_b]
        return self.root.codebooks_targets_[idxs, :]

    def get_params(self):
        kw = super().get_params()
        return kw.update({
            "root__m_": self.root.m_,
            "root__codebooks_": self.root.codebooks_,
            "root__codebooks_targets_": self.root.codebooks_targets_,
        })


class LVQ(BaseModel):
    """Learning Vector Quantization algorithm
    
    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param float learning_rate: learning rate
    :param int n_codebooks: how many codebooks to use
    :var np.ndarray codebooks_: codebooks
    :var np.ndarray codebooks_targets_: codebooks_targets
    """

    def __init__(self, loss=accuracy, learning_rate=0.0001, *args, **kwargs):
        self.n_codebooks_ = kwargs.pop('n_codebooks', None)
        super(LVQ, self).__init__(*args, **kwargs)
        self.learning_rate_ = learning_rate
        self.loss = loss
        self.solver_ = LVQSolver(self)

    def clean_copy(self):
        return LVQ(loss=self.loss, learning_rate=self.learning_rate_, n_codebooks=self.n_codebooks_)