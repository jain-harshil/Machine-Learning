from sklearn import tree
import pandas as pd
import random 
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        if criterion=="information_gain":
            self.criterion = "entropy"
        else:
            self.criterion = "gini"
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        self.clfs =[]
        self.clfsy = []
        self.trees = list()
        col = len(X.columns)
        for i in range(self.n_estimators):
            send = pd.concat([X,y.rename('y')],axis=1)
            sample = self.subsample(send, 1)
            clf = tree.DecisionTreeClassifier(criterion=self.criterion,max_depth = self.max_depth,max_features=col//2)
            y = sample['y']
            X = sample.drop(['y'],axis=1)
            X.reset_index(inplace=True,drop=True)
            y.reset_index(inplace=True,drop=True)
            self.clfs.append(X)
            self.clfsy.append(y)
            clf.fit(X,y)
            self.trees.append(clf)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        X = pd.DataFrame(X)
        predictions = [self.bagging_predict(self.trees, X.iloc[i]) for i in range (len(X))]
        return(predictions)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invoke print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.
        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        plt.figure()
        for i in self.trees:
            plot_tree(i,filled=True)
            plt.show()

        print("Printing decision surfaces of decision trees")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(2, 3, _ + 1)
            x_min, x_max = self.clfs[_].iloc[:, 0].min() - 1, self.clfs[_].iloc[:, 0].max() + 1
            y_min, y_max = self.clfs[_].iloc[:, 1].min() - 1, self.clfs[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.trees[_].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.clfsy[_] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.clfs[_].loc[idx[0][i]][0], self.clfs[_].loc[idx[0][i]][1],c=color,cmap=plt.cm.RdBu, edgecolor='black', s=15)
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()

        # Figure 2
        print("Printing decision surface by combining the individual estimators")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:, 0].max() + 1
        y_min, y_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = config.Classifier_RF.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(self.y == i)
            for i in range (len(idx[0])):
                plt.scatter(self.X.loc[idx[0][i]][0], self.X.loc[idx[0][i]][1],c=color,cmap=plt.cm.RdBu, edgecolor='black', s=15)
        plt.suptitle("Decision surface by combining individual estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()


    def subsample(self, dataset,ratio):
        sample = pd.DataFrame()
        n_sample = round(len(dataset)*ratio)
        while(len(sample) < n_sample):
            index = random.randrange(len(dataset))
            sample = sample.append(dataset[index:index+1])
        return sample

    def bagging_predict(self, trees, row):
        row = [np.array(row)]
        predictions = [tree.predict(row) for tree in trees]
        predictions = [predictions[i][0] for i in range (len(predictions))]
        return max(set(predictions), key=predictions.count)



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        if criterion=="variance":
            self.criterion = "mse"
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        self.clfs =[]
        self.clfsy = []
        self.trees = list()
        col = len(X.columns)
        for i in range(self.n_estimators):
            send = pd.concat([X,y.rename('y')],axis=1)
            sample = self.subsample(send, 1)
            clf = tree.DecisionTreeRegressor(criterion=self.criterion,max_depth = self.max_depth,max_features=col//2)
            y = sample['y']
            X = sample.drop(['y'],axis=1)
            X.reset_index(inplace=True,drop=True)
            y.reset_index(inplace=True,drop=True)
            self.clfs.append(X)
            self.clfsy.append(y)
            clf.fit(X,y)
            self.trees.append(clf)

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        X = pd.DataFrame(X)
        predictions = [self.bagging_predict(self.trees, X.iloc[i]) for i in range (len(X))]
        return(predictions)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """

        plt.figure()
        for i in self.trees:
            plot_tree(i,filled=True)
            plt.show()

        print("Printing decision surfaces of decision trees")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(2, 3, _ + 1)
            x_min, x_max = self.clfs[_].iloc[:, 0].min() - 1, self.clfs[_].iloc[:, 0].max() + 1
            y_min, y_max = self.clfs[_].iloc[:, 1].min() - 1, self.clfs[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.trees[_].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)            
            plt.scatter(self.clfs[_][0], self.clfs[_][1],c="black",cmap=plt.cm.RdBu, edgecolor='black', s=15)
            
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.axis("tight")

        plt.show()

        # Figure 2
        print("Printing decision surface by combining the individual estimators")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:, 0].max() + 1
        y_min, y_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = config.Regressor_RF.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
        plt.scatter(self.X[0], self.X[1],c="black",cmap=plt.cm.RdBu, edgecolor='black', s=15)
        plt.suptitle("Decision surface by combining individual estimators")
        #plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()

    def subsample(self, dataset,ratio):
        sample = pd.DataFrame()
        n_sample = round(len(dataset)*ratio)
        while(len(sample) < n_sample):
            index = random.randrange(len(dataset))
            sample = sample.append(dataset[index:index+1])
        return sample

    def bagging_predict(self, trees, row):
        row = [np.array(row)]
        predictions = [tree.predict(row) for tree in trees]
        predictions = [predictions[i][0] for i in range (len(predictions))]
        return sum(predictions)/len(predictions)
