"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from sklearn import tree
from linearRegression.linearRegression import LinearRegression
import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

########### BaggingClassifier ###################

np.random.seed(42)
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

# criteria = 'information_gain'
# tree = DecisionTree(criterion=criteria)
tree = "regressor"
config.Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
config.Classifier_B.fit(X, y)
y_hat = config.Classifier_B.predict(X)
[fig1, fig2] = config.Classifier_B.plot()
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


# Making a datset having 2 outliers to show bagging overcomes misclassification

N = 100
P = 8
NUM_OP_CLASSES = 2
n_estimators = 5

# criteria = 'information_gain'
# tree = DecisionTree(criterion=criteria)

X = []
for i in range (1,9):
	for j in range (1,9):
		X.append([i,j])
y = []
for i in range (64):
	y.append(0)

l1 = [1,2,3,4,5,9,10,11,12,13,17,18,20,21,25,26,27,28,29,33,34,35,36,37,40]

for i in l1:
	y[i-1] = 1

X = pd.DataFrame(X)
y = pd.Series(y)

tree = "classifier"
config.Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
config.Classifier_B.fit(X, y)
y_hat = config.Classifier_B.predict(X)
[fig1, fig2] = config.Classifier_B.plot()
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))