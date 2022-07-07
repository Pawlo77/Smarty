# from datasets import load_data
# from categorical import OneHotEncoder

# URL = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"

# ds = load_data(filename="iris.txt")
# ds.info()
# ds.head(10)

# target = ds.drop_c(name="species")
# target.info()
# target.head()

# en = OneHotEncoder().fit(target["species"])
# target["species"] = en.encode(target["species"])

from smart.datasets import load_data
from smart.models.linear import LinearRegression
from smart.models.api import evaluate_model
from smart.models.dumb import evaluate_dumb
from smart.config import set_config

# set_config("VERBOSE", False)

ds = load_data("auto-insurance.csv", header=False)
ds.set_target_classes(1)

lr = LinearRegression(learning_rate=0.001)
# lr.fit(ds, batch_size=32, epochs=5, repeat=5)
# print(lr.coefs_, lr.bias_)
# y_pred = lr.predict(ds)

scores = evaluate_model(ds, lr, batch_size=32, epochs=25, repeat=25, sampling_method="cross_val", drop_reminder=False)
print(scores)
# 
# print(evaluate_dumb(ds))