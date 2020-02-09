"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from linearRegression.linearRegression import LinearRegression
from sklearn.datasets import load_iris

import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

# np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
y = []
for i in range(N):
    y.append(random.choice([1,-1]))

X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(y, dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria, max_depth=1)
config.Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
config.Classifier_AB.fit(X, y)
y_hat = config.Classifier_AB.predict(X)
config.Classifier_AB.plot()
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features

target = load_iris()['target']
target[:100] = -1
target[100:] = 1
df = pd.DataFrame(load_iris()['data'])
df = df.rename(columns={1: "SW",3:"PW"})

df['label'] = target

df = df.sample(frac=1).reset_index(drop=True) # Shuffling the dataset

weight = np.ones(len(df))/len(df)

X = df[["SW","PW"]]
X = X.rename( columns={"SW": 0, "PW": 1})
y = pd.Series(df["label"],dtype="category")
X_train = X[:90] # 60% training dataset
y_train = y[:90]
X_test = X[90:] # 40% testing dataset
y_test = y[90:]

criteria = 'information_gain'
n_estimators = 3

tree = DecisionTree(criterion=criteria, max_depth=1)
clf = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators)
clf.fit(X, y)
y_hat = clf.predict(X_test)
clf.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))