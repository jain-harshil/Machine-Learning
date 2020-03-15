import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures

# x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
# y = 4*x + 7 + np.random.normal(0,3,len(x))

# print(len(x))

def normal(X,y):
    X_t = np.transpose(X)
    first = np.linalg.inv(X_t.dot(X))
    second = X_t.dot(y)
    return first.dot(second)

a = 0

l1 = [[],[],[],[],[]]
for degree in [1,3,5,7,9]:
    include_bias = True
    poly = PolynomialFeatures(degree,include_bias = include_bias)
    l = []
    for N in range (10,100):

        x = np.array([i*np.pi/180 for i in range(N,300,4)])
        y = 4*x + 7 + np.random.normal(0,3,len(x))
        X = poly.transform(x)

        theta = normal(X,y)
        #print(theta)
        l1[a].append(np.linalg.norm(np.array(theta)))
    a = a+1
N = [i for i in range (10,100)]

for i in range (5):
    plt.plot(N,l1[i],label = 'degree_'+str(2*i+1))
    plt.xlabel("Value of N")
    plt.ylabel("Magnitude of theta")
plt.legend(loc = 'best')
plt.show()

    # for i in range(len(theta)):
    #     if include_bias == True:
    #         print('theta'+str(i)+' is ',theta[i])
    #     else:
    #         print('theta'+str(i+1)+' is ',theta[i])


    # xlabel = []
    # ylabel = theta[:]
    # if include_bias == True:
    #     for i in range (len(theta)):
    #         xlabel.append("theta_"+str(i))
    # else:
    #     for i in range (len(theta)):
    #         xlabel.append("theta_"+str(i+1))
    # for i in range (len(theta)):
    #     ylabel[i] = theta[i]
    # plt.bar(xlabel,ylabel)   
    # plt.ylabel("Value of co-efficients")
    # plt.title("Bar plot showing the coefficients of theta")
    # plt.show()