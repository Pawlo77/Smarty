import inspect

from datasets import DataSet
from errors import assertion


# pretty display calculated confision matrix
def display_conf(conf, classes, display_names=False):
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
def prepare_ds(mode="supervised"):
    def wrapper(func):
        def prepare(*args, **kwargs):
            args = inspect.getcallargs(func, *args, **kwargs)
            kwargs = args.pop("kwargs")

            assertion(isinstance(args["ds"], DataSet), "Model entry must be a datasets.DataSet")
            if mode == "supervised":
                assertion(args["ds"].target_classes_ is not None, "DataSet without specified target.")

            if "batch_size" in kwargs:
                args["ds"].batch(kwargs["batch_size"])
            if "repeat" in kwargs:
                args["ds"].repeat(kwargs["repeat"])
            if "shuffle" in kwargs:
                args["ds"].shuffle(kwargs["shuffle"]) 

            return func(**args, **kwargs)
        return prepare
    return wrapper
    
def print_epoch(epoch, max_epoch, initial_message="train"):
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
    print(f"\r\tStep {step}/{max_step}", end="")

    if kwargs:
        print(" - ", end="")
        for name, val in kwargs.items():
            print(f"{name}: {val}, ", end="")

    if step == max_step:
        print("\n\n", end="")