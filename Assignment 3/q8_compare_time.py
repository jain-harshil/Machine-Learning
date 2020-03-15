import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

np.random.seed(42)

a = []
b = []
c = []
for i in range (10,2000):
	print(i)
	N = 15

	X = pd.DataFrame(np.random.randn(N, i))
	y = pd.Series(np.random.randn(N))

	LR = LinearRegression(fit_intercept=True)
	c.append(i)
	start_time = time.time()
	LR.fit_vectorised(X, y,15) 
	end_time = time.time()

	a.append(end_time-start_time)

	start_time = time.time()
	LR.fit_normal(X, y) 
	end_time = time.time()

	b.append(end_time-start_time)

plt.plot(c,a,label = 'Gradient Descent')
plt.plot(c,b,label = 'Normal Equation')
plt.legend(loc = 'best')
plt.show()


a = []
b = []
c = []
for i in range (10,10000):
	print(i)
	P = 15

	X = pd.DataFrame(np.random.randn(i, 15))
	y = pd.Series(np.random.randn(i))

	LR = LinearRegression(fit_intercept=True)
	c.append(i)
	start_time = time.time()
	LR.fit_vectorised(X, y,15) 
	end_time = time.time()

	a.append(end_time-start_time)

	start_time = time.time()
	LR.fit_normal(X, y) 
	end_time = time.time()

	b.append(end_time-start_time)

plt.plot(c,a,label = 'Gradient Descent')
plt.plot(c,b,label = 'Normal Equation')
plt.legend(loc = 'best')
plt.show()

# print('The time taken for standard gradient descent implementation is',end_time - start_time)

# LR = LinearRegression(fit_intercept=True)

# start_time = time.time()
# LR.fit_normal(X, y) 
# end_time = time.time()

# print('The time taken for normal equation implementation is',end_time - start_time)