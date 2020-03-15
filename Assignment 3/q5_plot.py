import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from metrics import *

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

def normal(X,y):
    X_t = np.transpose(X)
    first = np.linalg.inv(X_t.dot(X))
    second = X_t.dot(y)
    return first.dot(second)

l = []
for degree in [1,2,3,4,5,6,7,8,9]:
    include_bias = True
    poly = PolynomialFeatures(degree,include_bias = include_bias)
    X = poly.transform(x)
    theta = normal(X,y)

    # for i in range(len(theta)):
    #     if include_bias == True:
    #         print('theta'+str(i)+' is ',theta[i])
    #     else:
    #         print('theta'+str(i+1)+' is ',theta[i])
    l.append(np.linalg.norm(np.array(theta)))

a = [1,2,3,4,5,6,7,8,9]
b = l
plt.plot(a,b)
plt.xlabel("Degree of fitted polynmial")
plt.ylabel("Magnitude of theta")
plt.show()