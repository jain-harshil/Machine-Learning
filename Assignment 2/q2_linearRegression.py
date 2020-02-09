import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

from metrics import *

import seaborn as sns

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y) 
    y_hat = LR.predict(X)
    LR.plot()

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

a = []
for i in range (20):
	a.append([])

for N in range (50,70):
	for P in range (5,15):
		X = pd.DataFrame(np.random.randn(N, P))
		y = pd.Series(np.random.randn(N))
		LR = LinearRegression(fit_intercept=True)
		start_time = time.time()
		LR.fit(X, y)
		a[N-50].append(time.time()-start_time)

a = []
b = []

for N in range (30,50000):
	X = pd.DataFrame(np.random.randn(N, 10))
	y = pd.Series(np.random.randn(N))
	LR = LinearRegression(fit_intercept=True)
	start_time = time.time()
	LR.fit(X, y)
	a.append((time.time()-start_time)*(10**5))
	b.append(N)

plt.plot(b,a)
plt.xlabel("N")
plt.ylabel("Time taken for fitting times 10^5")
plt.show()

a = []
b = []

for P in range (50,1000):
	X = pd.DataFrame(np.random.randn(100, P))
	y = pd.Series(np.random.randn(100))
	LR = LinearRegression(fit_intercept=True)
	start_time = time.time()
	LR.fit(X, y)
	a.append((time.time()-start_time)*(10**2))

plt.plot(b,a)
plt.xlabel("P")
plt.ylabel("Time taken for fitting times 100")
plt.show()