import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *


def kfold_cross_validation(dataset,y):
    for i in range(5):
        test = dataset[82*i:82*i+82]
        ygr = y[82*i:82*i+82]
        if i==0:
        	train = dataset[82*i+82:]
        	label = y[82*i+82:]
        else:
        	train1 = dataset[0:82*i]
        	label1 = y[0:82*i]
        	train2 = dataset[82*i+82:]
        	label2 = y[82*i+82:]
        	train = np.append(train1,train2,axis=0)
        	label = label1.append(label2)
        clf = LinearRegression(fit_intercept=True)
        train = pd.DataFrame(train)
        label = pd.Series(label)
        test = pd.DataFrame(test)
        clf.fit(train,label)	
        yhat = clf.predict(test)
        clf.plot1()
        print("MAE for Fold "+str(i+1)+": ",end = "")
        print(mae(yhat,ygr))
        print("RMSE for Fold "+str(i+1)+": ",end = "")
        print(rmse(yhat,ygr))

X = pd.read_excel('realestate.xlsx')
X = X.drop(['No'],axis = 1)
X = X.dropna()
attb = list(X.columns)
attb.remove('y')
y = X['y']
X = X.drop(['y'],axis = 1)
kfold_cross_validation(X,y)
# LR = LinearRegression(fit_intercept=True)
# LR.fit(X[:310], y[:310]) 
# y_hat = LR.predict(X[310:])
# print(mae(y_hat,y[310:]))