import inspect
from functools import wraps

from smarty.datasets import DataSet
from smarty.errors import assertion
from smarty.config import get_config


# pretty display calculated confision matrix
def display_conf(conf, classes, display_names=False):
    """Displays the calculated confusion matrix conf

    :param list of lists | 2D np.ndarray conf: the confusion matrix
    :param list | 1D np.ndarray classes: classes names, where ith name is linked to ith column of confusion matrix
    :param bool display_names: whether to display names provided in classes
    """
    print("   " + "".join(f"{val : <4}" for val in range(len(classes))))
    for i in range(len(classes)):
        print(f"{i : <3}" + "".join(f"{val : <4}" for val in conf[i]))
    
    if display_names:
        print()
        print(" id -> name")
        for i, name in enumerate(classes):
            print(f"{i: >3} -> {name}")

# check the data for each model action
# X - DataSet with specified target_classes_ or np.ndarray
# y - np.ndarray, required if X is also np.ndarray and mode is "supervised"
# mode - supervised or not (any other will be treated as unsupervised)
def prepare_ds(mode="supervised", **super_kwargs):
    def wrapper(func):
        @wraps(func)
        def prepare(*args, **kwargs):
            rest = inspect.getcallargs(func, *args, **kwargs)
            kwargs = rest.pop("kwargs")
            args = rest.pop("args")

            assertion(isinstance(rest["ds"], DataSet), "Model entry must be a datasets.DataSet")
            if mode == "supervised":
                assertion(rest["ds"].target_classes_ is not None, "DataSet without specified target.")

            if "batch_size" in kwargs:
                rest["ds"].batch(kwargs["batch_size"])
            if "repeat" in kwargs:
                rest["ds"].repeat(kwargs["repeat"])
            if "shuffle" in kwargs:
                rest["ds"].shuffle(kwargs["shuffle"]) 
            if "drop_reminder" in kwargs:
                rest["ds"].drop_reminder_ = kwargs["drop_reminder"]
            
            batch_size = rest["ds"].batch_size_
            if "batch_size" in super_kwargs:
                rest["ds"].batch(super_kwargs["batch_size"])

            if mode == "prediction":
                shuffle = rest["ds"].shuffle_
                rest["ds"].shuffle(False)

            out = func(*args, **rest, **kwargs)

            if mode == "prediction":
                rest["ds"].shuffle(shuffle)
            rest["ds"].batch(batch_size)
            
            return out
        return prepare
    return wrapper
    
def print_epoch(epoch, max_epoch, initial_message="train"):
    if not get_config("VERBOSE"):
        return

    if epoch == 1:
        print("<*>" * 15)

    if epoch == 1 and initial_message == "train": 
        print(f"Starting training model for {max_epoch} epochs.")
    elif epoch == 1 and initial_message == "test":
        print(f"Starting making predictions.")
    elif epoch == 1: # custom message
        print(initial_message)

    print(f"Epoch {epoch}/{max_epoch}")

# model training utility function to check its performance
def print_step(step, max_step, *args, **kwargs):
    if not get_config("VERBOSE"):
        return

    print(f"\r\tStep {step}/{max_step}", end="")
    if kwargs:
        print(" - ", end="")
        for name, val in kwargs.items():
            print(f"{name}: {val}, ", end="")

    if step == max_step:
        print("\n\n", end="")

def print_info(info):
    if not get_config("VERBOSE"):
        return
    print(info)

# calls each callback found
def handle_callbacks(root, kwargs, **mykwargs):
    flag = True # marks whereas training can be continued

    if "callbacks" in kwargs:
        for callback in kwargs["callbacks"]:
            flag = flag and callback(root, **mykwargs)
        print()
    return flag