import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *



N = 15
P = 1
X = np.random.rand(N, P)
y = 4*X+7
X = pd.DataFrame(X)
y = y.reshape(len(y),)
y = pd.Series(y)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y,batch_size = 30,lr = 0.05,n_iter = 500) 
LR.save_figs(X,y)

