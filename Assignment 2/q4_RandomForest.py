"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

########### RandomForestClassifier ###################

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['information_gain', 'gini_index']:
    config.Classifier_RF = RandomForestClassifier(6, criterion = criteria)
    config.Classifier_RF.fit(X, y)
    y_hat = config.Classifier_RF.predict(X)
    config.Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

criteria = "variance"
config.Regressor_RF = RandomForestRegressor(6, criterion = criteria)
config.Regressor_RF.fit(X, y)
y_hat = config.Regressor_RF.predict(X)
config.Regressor_RF.plot()
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
