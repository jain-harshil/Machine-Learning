import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept=fit_intercept
        self.method = method
        global y_hat

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        if self.fit_intercept == True:
            self.y = y
            X = X.to_numpy()
            y = y.to_numpy()
            X_shape = X.shape[0]
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            X_transpose = np.transpose(X)
            X_tr_dot_x = X_transpose.dot(X)
            temp1 = np.linalg.pinv(X_tr_dot_x)
            temp2 = X_transpose.dot(y)
            self.theta = temp1.dot(temp2)
            self.theta_size = len(self.theta)
        elif self.fit_intercept == False:
            self.y = y
            X = X.to_numpy()
            y = y.to_numpy()
            X_transpose = np.transpose(X)
            X_tr_dot_x = X_transpose.dot(X)
            temp1 = np.linalg.pinv(X_tr_dot_x)
            temp2 = X_transpose.dot(y)
            self.theta = temp1.dot(temp2)
            self.theta_size = len(self.theta)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if self.fit_intercept == True:
            X = X.to_numpy()
            X_shape = X.shape[0]
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            y = X.dot(self.theta)
            self.y_hat = pd.Series(y)
            return pd.Series(y)
        elif self.fit_intercept == False:
            X = X.to_numpy()
            y = X.dot(self.theta)
            self.y_hat = pd.Series(y)
            return pd.Series(y)

    def plot(self):
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(yhat)
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (theta_i)

        """
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.y_hat,self.y,'ro')
        plt.xlabel("y_hat")
        plt.ylabel("y")
        plt.title("Scatter plot between y_hat and y")
        plt.subplot(2,2,2)
        xlabel = []
        ylabel = self.theta[:]
        if self.fit_intercept == True:
            for i in range (self.theta_size):
                xlabel.append("theta_"+str(i))
        else:
            for i in range (self.theta_size):
                xlabel.append("theta_"+str(i+1))
        for i in range (self.theta_size):
            ylabel[i] = self.theta[i]
        plt.bar(xlabel,ylabel,log=True)   
        plt.ylabel("Value of co-efficients in log scale")
        plt.title("Bar plot on a log scale showing the coefficients of theta")
        plt.show()

        a = []
        for i in range (len(self.y_hat)):
            a.append(self.y[i]-self.y_hat[i])
        b = a[:]
        a = pd.DataFrame(a)
        ax = a.plot.kde()
        b = pd.Series(b)
        mean = b.mean()
        var = b.var()
        print(mean,var)
        plt.xlabel("Residuals")
        plt.ylabel("Probability Density")
        plt.legend(["Output Class"])
        plt.title("KDE plot of residuals"+", mean = "+str(mean)+", variance = "+str(var))
        plt.show()

    def plot1(self):
        xlabel = []
        ylabel = self.theta[:]
        for i in range (self.theta_size):
            xlabel.append("theta_"+str(i))
        for i in range (self.theta_size):
            ylabel[i] = self.theta[i]
        plt.bar(xlabel,ylabel)   
        plt.ylabel("Value of co-efficients")
        plt.title("Bar plot showing the coefficients of theta")
        plt.show()

