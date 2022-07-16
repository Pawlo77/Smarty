import numpy as np

from smarty.models.utils import print_info


class EarlyStoping:
    """Early stopping callback
    To implement custom rule, re-implement get_score method

    :param int patience: Number of epochs to wait before stopping (from last impovement)
    :param str mode: 'max' - maxes loss, 'min' - minimalizes loss
    :param bool retrive_best: Whether to overwrite models params with best one found
    :param float min_delta: minimum loss change to be treated as imporvement
    """

    def __init__(self, patience=5, mode='min', retrive_best=True, min_delta=0.0001):
        self.patience_ = patience
        self.mode_ = mode
        self.retrive_best_ = retrive_best
        self.best_params = self.best_score = None
        self.epochs = 0 # number of epochs scince last imporvement
        self.min_delta = min_delta

    def get_score(self, losses):
        """Returns mean of losses
        
        :param list losses: list of each each batch loss from gradient descent
        """
        return np.mean(np.array(losses))

    def __call__(self, root, losses):
        score = self.get_score(losses)
        self.epochs += 1

        if self.best_score is None or (self.mode_ == "max" and score - self.min_delta > self.best_score) or (self.mode_ == "min" and score + self.min_delta < self.best_score):
            print_info(f"Improvement from {self.best_score} to {score}.")
            self.best_score = score
            self.best_params = root.get_params()
            self.epochs = 0
            return True # return Flag that here we allow further training

        elif self.epochs >= self.patience_:
            print_info(f"Early stopping (best score {self.best_score}).")
            if self.retrive_best_:
                root.set_params(self.best_params) # update model's parameters to best one found
            return False # return Flag that here we do not allow further training
        
        else:
            print_info(f"Score did not improved from {self.best_score}.")
            return True # return Flag that here we allow further training

