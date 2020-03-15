import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 10
P = 3
X = np.random.randint(15,size = (N, P))

X = pd.DataFrame(X)
X.columns = ['a','b','c']

t = X['a']*5

X['d'] = t

X = np.array(X)
y = np.random.randint(10,size = (N,))

#print(X)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y,10)
y_hat = LR.predict(X)
y_hat = y_hat/10**160

print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))