
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions


# Discrete Input discrete output case

discrete1 = []
discrete2 = []
for m in range (2,10):
	for n in range (100,120):
		X = pd.DataFrame({i:pd.Series(np.random.randint(m, size = n), dtype="category") for i in range(5)})
		y = pd.Series(np.random.randint(m, size = n),dtype="category")
		clf = DecisionTree("information_gain",max_depth=3)
		start_time = time.time()
		clf.fit(X,y)
		discrete1.append(time.time()-start_time)
		start_time = time.time()
		yhat = clf.predict(X)
		discrete2.append(time.time()-start_time)
plt.plot(discrete1)
plt.xlabel("Iteration number")
plt.ylabel("Time taken for fitting real input discrete output")
plt.show()

plt.xlabel("Iteration number")
plt.ylabel("Time taken for predicting real input discrete output")
plt.plot(discrete2)
plt.show()

# real input discrete output case
discrete1 = []
discrete2 = []
for m in range (5,10):
	for n in range (100,120):
		X = pd.DataFrame(np.random.randn(m, n))
		y = pd.Series(np.random.randint(n, size = m), dtype="category")
		clf = DecisionTree("gini_index",max_depth = 2)
		start_time = time.time()
		clf.fit(X,y)
		discrete1.append(time.time()-start_time)
		start_time = time.time()
		yhat = clf.predict(X)
		discrete2.append(time.time()-start_time)
print(discrete1)
plt.plot(discrete1)
plt.xlabel("Iteration number")
plt.ylabel("Time taken for fitting real input discrete output")
plt.show()

plt.xlabel("Iteration number")
plt.ylabel("Time taken for predicting real input discrete output")
plt.plot(discrete2)
plt.show()

# discrete input real output
real1 = []
real2 = []
for m in range (5,10):
	for n in range (100,120):
		X = pd.DataFrame({i:pd.Series(np.random.randint(m, size = n), dtype="category") for i in range(5)})
		y = pd.Series(np.random.randn(n))
		clf = DecisionTree("criterion",max_depth = 4)
		start_time = time.time()
		clf.fit(X,y)
		real1.append(time.time()-start_time)
		start_time = time.time()
		yhat = clf.predict(X)
		real2.append(time.time()-start_time)
print(real1)
plt.plot(real1)
plt.xlabel("Iteration number")
plt.ylabel("Time taken for fitting discrete input real output")
plt.show()

plt.xlabel("Iteration number")
plt.ylabel("Time taken for predicting discrete input real output")
plt.plot(real2)
plt.show()


# real input real output

real1 = []
real2 = []
for m in range (5,10):
	for n in range (100,120):
		X = pd.DataFrame(np.random.randn(n, m))
		y = pd.Series(np.random.randn(n))
		clf = DecisionTree("criterion",max_depth = 4)
		start_time = time.time()
		clf.fit(X,y)
		real1.append(time.time()-start_time)
		start_time = time.time()
		yhat = clf.predict(X)
		real2.append(time.time()-start_time)
print(real1)
plt.plot(real1)
plt.xlabel("Iteration number")
plt.ylabel("Time taken for fitting real input real output")
plt.show()

plt.xlabel("Iteration number")
plt.ylabel("Time taken for predicting real input real output")
plt.plot(real2)
plt.show()
