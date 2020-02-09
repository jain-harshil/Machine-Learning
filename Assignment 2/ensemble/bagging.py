from sklearn import tree
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import config

import logging
logging.getLogger().setLevel(logging.CRITICAL)

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        self.clfs =[]
        self.clfsy = []
        self.trees = list()
        for i in range(self.n_estimators):
            send = pd.concat([X,y.rename('y')],axis=1)
            sample = self.subsample(send, 1)
            if self.base_estimator == "classifier":
                clf = tree.DecisionTreeClassifier(max_depth=10)
            elif self.base_estimator == "regressor":
                clf = tree.DecisionTreeRegressor(max_depth=10)
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
        Funtion to run the BaggingClassifier on a data point
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
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
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
        Z = config.Classifier_B.predict(np.c_[xx.ravel(), yy.ravel()])
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
        fig2 = plt

        return [fig1,fig2]

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