import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from copy import copy
from matplotlib.colors import ListedColormap

import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.clfs = []

        self.initial_weights = np.array([])
        self.arr_weight = list()
        self.pred_fi = list()
        self.arr_alpha = list()

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y

        self.arr_weight = []
        self.pred_fi = []
        self.arr_alpha = []
        wt = np.ones(len(y))/len(y)
        weights = np.copy(wt)
        self.initial_weights = np.copy(wt)
        gr_truth = np.array(y)
        self.arr_weight.append(np.copy(wt))

        for i in range(self.n_estimators):
            self.base_estimator.fit(X,y,self.arr_weight[i])
            y_pred = self.base_estimator.predict(X)

            predicted = np.array(y_pred)
            t = np.nonzero(predicted-gr_truth)
            error = sum(weights[(t[0])])
            error /= sum(weights)
            alpha_m = 0.5*np.log((1-error)/error) # Calculation of error
            self.arr_alpha.append(alpha_m)
            weights[np.nonzero(predicted - gr_truth)[0]]*=np.exp(alpha_m)
            factor = np.exp(-alpha_m)
            weights[np.delete(np.arange(len(y)),(np.nonzero(predicted - gr_truth)[0]))] = weights[np.delete(np.arange(len(y)),(np.nonzero(predicted - gr_truth)[0]))]*factor
            norm = sum(weights)
            weights = weights/norm # normalization

            self.clfs.append(copy(self.base_estimator))
            self.pred_fi.append(copy(predicted))
            self.arr_weight.append(copy(weights))
            
    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        ypred = np.zeros(len(X))

        for i in range (len(self.arr_alpha)):
            yhat = self.clfs[i].predict(X)
            ypred = ypred+self.arr_alpha[i]*pd.Series(yhat)

            if i == len(self.arr_alpha)-1:
                ypred = np.sign(ypred)

        return ypred

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        

        print("Printing decision surfaces of decision trees")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(2, 3, _ + 1)
            x_min, x_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:, 0].max() + 1
            y_min, y_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.clfs[_].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array(Z)
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
            for i, color in zip(range(n_classes), plot_colors):
                if i == 0:
                    idx = np.where(self.y == -1)
                if i == 1:
                    idx = np.where(self.y == 1)
                for i in range (len(idx[0])):
                    plt.scatter(self.X.loc[idx[0][i]][0], self.X.loc[idx[0][i]][1],c=color,cmap=plt.cm.RdBu, edgecolor='black', s=15)
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig1 = plt

        # Figure 2
        print("Printing decision surface by combining the individual estimators")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:, 0].max() + 1
        y_min, y_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = config.Classifier_AB.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
        for i, color in zip(range(n_classes), plot_colors):
            if i == 0:
                idx = np.where(self.y == -1)
            if i == 1:
                idx = np.where(self.y == 1)
            for i in range (len(idx[0])):
                plt.scatter(self.X.loc[idx[0][i]][0], self.X.loc[idx[0][i]][1],c=color,cmap=plt.cm.RdBu, edgecolor='black', s=15)
        plt.suptitle("Decision surface by combining individual estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig2 = plt

        return [fig1,fig2]
