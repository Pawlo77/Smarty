from smart.datasets import DataSet, train_test_split, cross_val_split
from smart.errors import assertion
from .metrics import mean_squared_error


SAMPLINGS_METHODS = (
    "train_test_split",
    "cross_val",
)

# accepts an unfitted model, trains it according to sampling_method and returns calculated metric
def evaluate_model(ds, model, metric=mean_squared_error, sampling_method="train_test_split", *args, **kwargs):
    assertion(sampling_method in SAMPLINGS_METHODS, "Sampling method not recognised, see models.api.SAMPLING_METHODS to see available one.")    

    if sampling_method == "train_test_split":
        return _evaluate_train_test_split(ds, model, metric, *args, **kwargs)
    else: # "cross_val"
        return _evaluate_cross_val_test_split(ds, model, metric, **kwargs)

def _evaluate_cross_val_test_split(ds, model, metric, *args, **kwargs):
    # by default drop reminder for cross_validation, which if relatively small might strongly affect model's score
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
        y_pred = cur_model.predict(test_ds, *args, **kwargs)
        scores.append(metric(test_ds.get_target_classes(), y_pred))
    return scores

def _evaluate_train_test_split(ds, model, metric, *args, **kwargs):
    train_ds, test_ds = train_test_split(ds, *args, **kwargs)
    history = model.fit(train_ds, *args, **kwargs)
    y_pred = model.predict(test_ds, *args, **kwargs)
    return metric(test_ds.get_target_classes(), y_pred)