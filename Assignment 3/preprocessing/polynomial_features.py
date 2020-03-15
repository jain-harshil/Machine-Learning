''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        
        self.degree = degree
        self.include_bias = include_bias
    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        # X = list(X)
        # a = []
        # if self.include_bias == True:
        #     a.append(1)
        # for j in range (self.degree):
        #     for i in range (len(X)):
        #         a.append(X[i]**(j+1))
        # a = np.array(a)
        # return a

        new_array = np.ones(len(X))
        new_array = new_array[:,np.newaxis]
        for i in range(1,self.degree+1):
            temp_arr = X**i
            temp_arr = temp_arr[:,np.newaxis]
            new_array = np.c_[ new_array, temp_arr ]    
        X = new_array
        if self.include_bias == False:
            X = X[:,1:]
        return X