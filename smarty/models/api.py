from smarty.datasets import train_test_split, cross_val_split
from smarty.errors import assertion


SAMPLINGS_METHODS = (
    "train_test_split",
    "cross_val",
)
"""Available samplings method for auto model evaluation"""

# accepts an unfitted model, trains it according to sampling_method and returns calculated score
def evaluate_model(ds, model, sampling_method="train_test_split", *args, **kwargs):
    """Splits ds and trains the model according to sampling_method, than evaluates it also according to sampling_method

    :param DataSet ds: a DataSet - data source
    :param class model: unfitted model, has to have .fit(), .predict(), .clean_copy() methods
    :param function metric: evaluation mertic, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param str sampling_method: one of smarty.models.api.SAMPLINGS_METHODS
    :params args, kwargs: additional params that will be passed to sampling function and model methods
    :raises: AssertionError if sampling_method not recognized
    :returns: score or list of scores

    .. note::
        If drop_reminder not passed in kwargs, cross_val_split by default will drop last fold if not full
    """

    assertion(sampling_method in SAMPLINGS_METHODS, "Sampling method not recognised, see models.api.SAMPLING_METHODS to see available one.")    

    if sampling_method == "train_test_split":
        return _evaluate_train_test_split(ds, model, *args, **kwargs)
    else: # "cross_val"
        return _evaluate_cross_val_test_split(ds, model, **kwargs)

def _evaluate_cross_val_test_split(ds, model, *args, **kwargs):
    if "drop_reminder" not in kwargs: 
        kwargs["drop_reminder"] = True
    
    folds = cross_val_split(ds, *args, **kwargs)
    scores = []

    for i in range(len(folds)): # at each iteration different fold is marked as target one 
        test_ds = folds[i]
        cur_model = model.clean_copy() # copy original unfitted model
        training_indeces = list(range(len(folds)))
        training_indeces.remove(i) # drop current test_ds
        
        # join rest of folds together to create training_ds
        train_ds = folds[training_indeces[0]].copy()
        for idx in training_indeces[1:]:
            train_ds.join(folds[idx], axis=0)
        
        history = cur_model.fit(train_ds, *args, **kwargs)
        score = cur_model.evaluate(test_ds, *args, **kwargs)
        scores.append(score)
    return scores

def _evaluate_train_test_split(ds, model, *args, **kwargs):
    train_ds, test_ds = train_test_split(ds, *args, **kwargs)
    history = model.fit(train_ds, *args, **kwargs)
    return model.evaluate(test_ds, *args, **kwargs)