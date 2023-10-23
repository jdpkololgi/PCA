import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into a Pandas DataFrame

df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

from sklearn.preprocessing import StandardScaler

features = ['sepal length','sepal width','petal length','petal width','target']

# Separating out the features

x = df.loc[:, features].values

# Separating out the target

y = df.loc[:, ['target']].values

# Standardising the features

x = StandardScaler().fit_transform(x)

