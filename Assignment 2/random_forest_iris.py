import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForestiris import RandomForestClassifier
from tree.randomForestiris import RandomForestRegressor

from sklearn.datasets import load_iris

import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

###Write code here
np.random.seed(42)

dataset = load_iris()
X, yhat = dataset.data, dataset.target 
X = pd.DataFrame(X)
y = pd.Series(yhat)
y = y.astype("category")
conc = pd.concat([X,y.rename('y')],axis=1)
conc = conc.sample(frac=1).reset_index(drop=True)
y = conc['y']
X = conc.drop(columns = ['y'])
X = X.drop(columns = [0,2])
train = X[0:90]
trainlabel = y[0:90]
test = X[90:150]
testlabel = y[90:150]

train= train.rename({1: 0, 3: 1}, axis=1)
test= test.rename({1: 0, 3: 1}, axis=1)
config.Classifier_RF = RandomForestClassifier(6, criterion = "information_gain")
config.Classifier_RF.fit(train, trainlabel)

y_hat = config.Classifier_RF.predict(test)
config.Classifier_RF.plot()
print('Criteria :', "information_gain")
print('Accuracy: ', accuracy(y_hat, testlabel))
for cls in y.unique():
    print('Precision: ', precision(testlabel, y_hat, cls))
    print('Recall: ', recall(testlabel, y_hat, cls))

