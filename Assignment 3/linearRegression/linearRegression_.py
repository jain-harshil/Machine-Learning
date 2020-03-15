# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here

import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad


class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.theta_history = None
        self.thetas = []
        self.X = None
        self.y = None

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if self.fit_intercept == True:
            theta = np.random.randn(X.shape[1]+1,1) 
            X = np.array(X)
            X_shape = X.shape[0]   
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            y = np.array(y)
            y = y.reshape(len(y),1)
            m = len(y)
            k = X.shape[1]
            cost_history = np.zeros(n_iter)
            # theta_history = np.zeros((n_iter,k))
            for it in range (n_iter):
                if lr_type == 'constant':
                    m1 = 1
                elif lr_type == 'inverse':
                    m1 = 1/(it+1)
                mini_batches = self.create_mini_batches(X,y,batch_size)
                for mini_batch in mini_batches:
                    X,y = mini_batch
                    prediction = np.dot(X,theta)
                    for i in range (len(theta)):
                        r = np.array(X[:,i])
                        r = r.reshape(len(r),1)
                        theta[i] = theta[i] - (2/m)*lr*m1*(r.T.dot(prediction-y))
                    #theta_history[it,:] = theta.T
                    cost_history[it] = self.cost(theta,X,y)
            self.coef_ = theta
        elif self.fit_intercept == False:
            theta = np.random.randn(X.shape[1],1)
            X = np.array(X)
            y = np.array(y)
            y = y.reshape(len(y),1)
            m = len(y)
            k = X.shape[1]
            cost_history = np.zeros(n_iter)
            # theta_history = np.zeros((n_iter,k))
            for it in range (n_iter):
                if lr_type == 'constant':
                    m1 = 1
                elif lr_type == 'inverse':
                    m1 = 1/(it+1)
                mini_batches = self.create_mini_batches(X,y,batch_size)
                for mini_batch in mini_batches:
                    X,y = mini_batch
                    prediction = np.dot(X,theta)
                    for i in range (len(theta)):
                        r = np.array(X[:,i])
                        r = r.reshape(len(r),1)
                        theta[i] = theta[i] - (2/m)*lr*m1*(r.T.dot(prediction-y))
                    # theta_history[it,:] = theta.T
                    cost_history[it] = self.cost(theta,X,y)
            self.coef_ = theta

    def fit_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if self.fit_intercept == True:
            #theta = np.random.randn(X.shape[1]+1,1) 
            theta = np.array([-2,8]).T
            theta = theta.reshape(len(theta),1)
            # print(theta)
            # print(theta.shape)
            theta = np.array(theta)
            X = np.array(X)
            X_shape = X.shape[0]   
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            self.X = X
            self.y = y
            y = np.array(y)
            y = y.reshape(len(y),1)
            m = len(y)
            k = X.shape[1]
            cost_history = np.zeros(n_iter)
            self.theta_history = np.zeros((n_iter,k))
            for it in range (n_iter):
                if lr_type == 'constant':
                    m1 = 1
                elif lr_type == 'inverse':
                    m1 = 1/(it+1)
                mini_batches = self.create_mini_batches(X,y,batch_size)
                for mini_batch in mini_batches:
                    X,y = mini_batch
                    prediction = np.dot(X,theta)
                    theta = theta - (2/m)*lr*m1*(X.T.dot((prediction-y)))
                    self.theta_history[it,:] = theta.T
                    self.thetas.append(theta)
                    cost_history[it] = self.cost(theta,X,y)
            self.coef_ = theta

        elif self.fit_intercept == False:
            theta = np.random.randn(X.shape[1],1)
            X = np.array(X)
            y = np.array(y)
            y = y.reshape(len(y),1)
            m = len(y)
            k = X.shape[1]
            cost_history = np.zeros(n_iter)
            # theta_history = np.zeros((n_iter,k))
            for it in range (n_iter):
                if lr_type == 'constant':
                    m1 = 1
                elif lr_type == 'inverse':
                    m1 = 1/(it+1)
                mini_batches = self.create_mini_batches(X,y,batch_size)
                for mini_batch in mini_batches:
                    X,y = mini_batch
                    prediction = np.dot(X,theta)
                    theta = theta - (2/m)*lr*m1*(X.T.dot((prediction-y)))
                    # theta_history[it,:] = theta.T
                    cost_history[it] = self.cost(theta,X,y)
            self.coef_ = theta

    def fit_autograd(self, X, y, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if self.fit_intercept == True:
            X = np.array(X)
            y = np.array(y)
            X_shape = X.shape[0]   
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            self.X = X
            self.y = y
            theta_old = np.random.randn(X.shape[1])
            agrad = elementwise_grad(self.cost_function)
            for i in range(n_iter):
                if lr_type == 'constant':
                    m1 = 1
                elif lr_type == 'inverse':
                    m1 = 1/(i+1)
                val = agrad(theta_old)
                predicted = X.dot(theta_old)
                error = y - predicted
                temp = theta_old - lr*m1*val
                theta_old = temp
            self.coef_ = theta_old
        elif self.fit_intercept == False:
            X = np.array(X)
            y = np.array(y)
            self.X = X
            self.y = y
            theta_old = np.random.randn(X.shape[1])
            agrad = elementwise_grad(self.cost_function)
            for i in range(n_iter):
                if lr_type == 'constant':
                    m1 = 1
                elif lr_type == 'inverse':
                    m1 = 1/(i+1)
                val = agrad(theta_old)
                predicted = X.dot(theta_old)
                error = y - predicted
                temp = theta_old - lr*m1*val
                theta_old = temp
            self.coef_ = theta_old

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        if self.fit_intercept == True:
            self.y = y
            if isinstance(X,pd.DataFrame):
                X = X.to_numpy()
            if isinstance(X,pd.Series):
                y = y.to_numpy()
            X_shape = X.shape[0]
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            X_transpose = np.transpose(X)
            X_tr_dot_x = X_transpose.dot(X)
            temp1 = np.linalg.inv(X_tr_dot_x)
            temp2 = X_transpose.dot(y)
            self.coef_ = temp1.dot(temp2)
            self.theta_size = len(self.coef_)
        elif self.fit_intercept == False:
            self.y = y
            if isinstance(X,pd.DataFrame):
                X = X.to_numpy()
            if isinstance(X,pd.Series):
                y = y.to_numpy()
            X_transpose = np.transpose(X)
            X_tr_dot_x = X_transpose.dot(X)
            temp1 = np.linalg.inv(X_tr_dot_x)
            temp2 = X_transpose.dot(y)
            self.coef_ = temp1.dot(temp2)
            self.theta_size = len(self.coef_)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if self.fit_intercept == True:
            if isinstance(X,pd.DataFrame):
                X = X.to_numpy()
            X_shape = X.shape[0]
            X_bias = np.ones((X_shape,1))
            X = np.append(X_bias,X,axis = 1)
            y = X.dot(self.coef_)
            y = y.reshape(len(y),)
            self.y_hat = pd.Series(y)
            return pd.Series(y)
        elif self.fit_intercept == False:
            if isinstance(X,pd.DataFrame):
                X = X.to_numpy()
            y = X.dot(self.coef_)
            y = y.reshape(len(y),)
            self.y_hat = pd.Series(y)
            return pd.Series(y)

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """


    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """



    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        # print(self.coef_)

    def cost(self, theta, X, y):
        X = np.array(X)
        y = np.array(y)
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/m) * np.sum(np.square(predictions-y))
        return cost

    def create_mini_batches(self, X, y, batch_size): 
        mini_batches = [] 
        data = np.hstack((X, y)) 

        n_minibatches = data.shape[0] // batch_size 

        i = 0
        for i in range(n_minibatches): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def cost_function(self,theta):
        error = ((self.y - np.dot(self.X,theta))**2)/len(self.X)
        return error

    def save_figs(self,X,y):
        X = list(np.array(X))
        y = list(y)
        x = []
        for i in range (len(X)):
            x.append(X[i][0])
        print(x)
        print(y)
        for j in range (49):
            c,m = self.theta_history[j*10]
            regression_line = []
            for i in x:
                regression_line.append((m*i)+c)
            plt.figure()
            plt.scatter(x,y,color='b')
            plt.xlabel("x")
            plt.ylabel("y")
            m = float("{0:.2f}".format(m))
            c = float("{0:.2f}".format(c))
            plt.plot(x,regression_line,'-r')
            plt.title("m = "+str(m)+' and '+"c = "+str(c))
            print(j)
            plt.savefig(str(j+1)+'.png')