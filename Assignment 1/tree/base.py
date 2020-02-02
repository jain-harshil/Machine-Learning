"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index, variance_gain, gini_gain, inf_gain
import math


np.random.seed(42)

def red(data,attri):
    var = {}
    for attr in attri:
        freq = {}
        for i in data[attr]:
            length = len(data[attr])
            if i not in freq:
                freq[i]=1/length
            else:
                freq[i] = freq[i]+1/length
        keys = list(freq.keys())
        st = 0
        for i in keys:
            temp = (data[data[attr]==i])
            st+=np.std(temp['y'])*len(temp)/len(data)
        var[attr]=st
    return min(var,key = var.get)

def dtreereg(dataset,data,attr,min1,parent):

    d1 = data['y']
    d2 = dataset['y']
    if len(data)<=min1:
        return d1.mean()
    elif len(attr)==0:
        return parent
    elif len(data)==0:
        return d2.mean()
    else:
        parent = d1.mean()
        best = red(data,attr)
        tree = {best:{}}
        attr.remove(best)
        base = [i for i in data[best].unique()]
        for j in base:
            n_data = data.where(data[best]==j).dropna()
            subtree = dtreereg(dataset,n_data,attr, min1,parent)
            tree[best][j] = subtree
    return tree

def pred_reg(tree,inputs):
        node = tree
        for i in node.keys():
            if i in inputs:
                try:
                    p = node[i][inputs[i]]
                except:
                    return df_reg['y'].mean()
                p = node[i][inputs[i]]  
                if type(p) == dict:
                    return pred_reg(p, inputs)
                else:
                    return p

class Node:
    def __init__(self, predicted_class):
        self.th = 0
        self.predicted_class = predicted_class
        self.f_index = 0
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self, criterion,max_depth = None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"}
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.input_type = None
        self.output_type = None

    def fit(self, X, y):
        # if isinstance (X,pd.DataFrame):
        #     X = X.to_numpy()
        # if isinstance(y,pd.Series):
        #     y = y.to_numpy()
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.input_type = X.dtypes[0].name 
        self.output_type = y.dtype.name

        # Real Input Discrete Output Case

        if X.dtypes[0].name != "category" and y.dtype.name == "category":
            self.num_classes = len(list(set(y)))
            self.num_features = X.shape[1]
            self.tree = self.tree1(X,y)

        # Discrete Input Discrete Output Case

        elif X.dtypes[0].name == "category" and y.dtype.name == "category":
            attributes = list(X.columns)
            X['Class'] = y
            self.tree = self.dtree(X,attributes,None)

        # Discrete Input Real Output Case

        elif X.dtypes[0].name == "category" and y.dtype.name != "category":
            attributes = list(X.columns)
            X['Class'] = y
            self.tree = self.dtree1(X,attributes,None)

        # Real Input Real Output Case

        elif X.dtypes[0].name != "category" and y.dtype.name != "category":

            y = y.rename("y")
            attrb = list(X.columns)
            df = pd.concat([X, y], axis=1, sort=False)
            self.tree = dtreereg(df,df,attrb,4,None)

    def split(self, X, y):

        # X = X.to_numpy()
        # y = y.to_numpy()

        X = pd.DataFrame(X)
        y = pd.Series(y)
        y = y.astype("category")

        #Real Input Discrete Output Case
        if X.dtypes[0].name != "category" and y.dtype.name == "category":

            X = X.to_numpy()
            y = y.to_numpy()
            size = y.size
            a = None
            b = None

            if size<2:
                return a,b

            bg = gini_index(y)

            b_id, b_th = None, None

            for j in range (self.num_features):
                th_arr, cl_arr = zip(*sorted(zip(X[:, j], y)))

                l = [0] * self.num_classes
                r = [np.sum(y == c) for c in range(self.num_classes)]

                for i in range (1,size):
                    c = cl_arr[i-1]
                    l[c] = l[c] + 1
                    r[c] = r[c] - 1
                    if self.criterion == "gini_index":
                        gini = (i* gini_index(l) +(size-i)*gini_index(r))/size
                    elif self.criterion == "information_gain":
                        gini = (i* inf_gain(l) +(size-i-1)*inf_gain(r))/(size+5)

                    if th_arr[i] == th_arr[i-1]:
                        continue

                    if gini<bg and self.criterion=="gini_index":
                        bg = gini
                        b_id = j
                        b_th = (th_arr[i]+th_arr[i-1])
                        b_th = b_th/2
                    if gini>bg and self.criterion=="information_gain":
                        bg = gini
                        b_id = j
                        b_th = (th_arr[i]+th_arr[i-1])
                        b_th = b_th/2
            return b_id,b_th

    #Real Input Discrete Output Case

    def tree1(self, X, y,depth = 2):

        #Real Input Discrete Output Case

        X = pd.DataFrame(X)
        y = pd.Series(y)
        y = y.astype("category")

        if X.dtypes[0].name != "category" and y.dtype.name == "category":

            X = X.to_numpy()
            y = y.to_numpy()
            n_per = [0]*(self.num_classes)
            for i in range (len(n_per)):
                n_per[i] = np.sum(y == i)
            predicted_class = np.argmax(n_per)
            node = Node(predicted_class = predicted_class)
            if depth<self.max_depth:

                id1, thr = self.split(X,y)
                if id1 is not None:
                    i_l = X[:,id1] < thr
                    Xr, yr = X[~i_l],y[~i_l]    
                    Xl, yl = X[i_l],y[i_l]
                    node.f_index = id1
                    node.th = thr
                    node.left = self.tree1(Xl,yl,depth+1)
                    node.right = self.tree1(Xr,yr,depth+1)
            return node

    

    # Discrete Input Discrete Output Case

    def dtree (self, X, attributes, parent):

        attribute = [i for i in attributes]

        z = X['Class']
        #If all target_values have the same value, return this value
        if len(z.unique())<=1:
            return (list(z.unique())[0])
        #If the dataset is empty, return the mode target feature value in the original dataset
        elif len(X)==0:
            return np.unique(z)[np.argmax(np.unique(z,return_counts=True)[1])]
        #If the feature space is empty, return the mode target feature value of the direct parent node 	
        elif len(attribute)==0:
            return parent
        #If none of the above holds true, grow the tree
        else:
            parent = np.unique(z)[np.argmax(np.unique(z,return_counts=True)[1])]
        a = []

        if self.criterion=="information_gain":
            for i in attribute:
                attr = X[i]
                a.append(information_gain(z,attr))
            best = attribute[a.index(max(a))]

        elif self.criterion=="gini_index":
            for i in attribute:
                attr = X[i]
                a.append(gini_gain(z,attr))
            best = attribute[a.index(min(a))]

        # best = info_gain(X['Class'],X[attribute],list(X.columns))
    
        # tree = {best:{}}
        # attribute.remove(best)
        
        # print(best)
        tree = {best:{}}
        attribute.remove(best)
        for ndata in X[best].unique():
            new_data = X[X[best]==ndata]
            # print(new_data)
            subtree = self.dtree(new_data,attribute,parent)
            tree[best][ndata]=subtree
        return tree  

    # Discrete Input Real Output Case

    def dtree1 (self, X, attributes, parent):

        attribute = [i for i in attributes]

        z = X['Class']
        if len(z.unique())<=1:
            return z.mean()
        elif len(X)==0:
            return z.mean()
        elif len(attribute)==0:
            return parent
        else:
            parent = z.mean()
        a = []

        for i in attribute:
            attr = X[i]
            a.append(variance_gain(z,attr))

        # best = info_gain(X['Class'],X[attribute],list(X.columns))
    
        # tree = {best:{}}
        # attribute.remove(best)
        best = attribute[a.index(max(a))]
        # print(best)
        tree = {best:{}}
        attribute.remove(best)
        for ndata in X[best].unique():
            new_data = X[X[best]==ndata]
            # print(new_data)
            subtree = self.dtree1(new_data,attribute,parent)
            tree[best][ndata]=subtree
        return tree  

    def predict(self, X):
        # X = X.to_numpy()
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

        # Real Input Discrete Output Case

        if self.input_type != "category" and self.output_type == "category":
            y_pred = np.zeros(len(X))
            for i in range (len(X)):
                val = self.subs(X.iloc[i])
                y_pred[i] = val
            return pd.Series(y_pred)

        elif self.input_type != "category" and self.output_type != "category":
            y_pred = np.zeros(len(X))
            for i in range (len(X)):
                val = pred_reg(self.tree,X.iloc[i])
                y_pred[i] = val
            return pd.Series(y_pred)

        # Discrete Input Discrete Output Case

        elif self.input_type == "category" and self.output_type == "category":
            y = []
            attri = list(X.columns)
            attri.remove('Class')
            for i in range (len(X)):
                a = self.tree
                c = X.iloc[i]
                list1 = list(a.keys())
                while(1):
                    a = a[list1[0]][c[list1[0]]]
                    list1.remove(list1[0])
                    if type(a)!=dict:
                        y.append(a)
                        break
                    elif (list(a.keys()))[0] in attri:
                        list1.append(list(a.keys())[0])
            y = pd.Series(y)
            return y

        # Discrete Input Real Output Case

        elif self.input_type == "category" and self.output_type != "category":
            y = []
            attri = list(X.columns)
            attri.remove('Class')
            for i in range (len(X)):
                a = self.tree
                c = X.iloc[i]
                list1 = list(a.keys())
                while(1):
                    a = a[list1[0]][c[list1[0]]]
                    list1.remove(list1[0])
                    if type(a)!=dict:
                        y.append(a)
                        break
                    elif (list(a.keys()))[0] in attri:
                        list1.append(list(a.keys())[0])
            y = pd.Series(y)
            return y

    def subs(self, inputs):
        node = self.tree
        while(node.left):
            if inputs[node.f_index]<node.th:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


    def plot(self):

        if self.input_type == "category" or (self.input_type != "category" and self.output_type != "category"):
            maindict = self.tree

            def printdict(d, indent=0):
                for key, value in d.items():
                    print('\t' * indent + str(key))
                    if isinstance(value, dict):
                        printdict(value, indent+1)
                    else:
                        print('\t' * (indent+1) + str(value))
            printdict(maindict)

        # def plothelp(node,depth=0):
        #     if isinstance(node, dict):
        #         print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        #         plothelp(node['left'], depth+1)
        #         plothelp(node['right'], depth+1)
        #     else:
        #         print('%s[%s]' % ((depth*' ', node.f_index)))
        # maindict = self.tree
        # plothelp(maindict)
        
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

# if __name__ == "__main__":

#     # Discrete Input and Discrete Output Case Test tennis.csv input

#     dt = pd.read_csv('tennis.csv')
#     dt = dt.drop(['Day'],axis = 1)
#     yhat = dt['Class'] 
#     dt = dt.drop(['Class'],axis = 1)
#     dt = dt.astype("category")
#     yhat = yhat.astype("category")
#     clf = DecisionTree("criterion",max_depth = 5)
#     clf.fit(dt,yhat)
#     y = clf.predict(dt)
#     clf.plot()
#     # from sklearn.datasets import load_iris
#     # dataset = load_iris()
#     # X, yhat = dataset.data, dataset.target 
#     # print(yhat) 
#     # X = pd.DataFrame(X)
#     # yhat = pd.Series(yhat)
#     # yhat = yhat.astype("category")
#     # clf = DecisionTree("criterion",max_depth = 5)
#     # clf.fit(X[:105],yhat[:105])
#     # X = X.to_numpy()
#     # y = clf.predict(X[105:])
#     # print(yhat[105:])
#     # print(y)
#     # print(accuracy(yhat[105:],y))

    # X = pd.DataFrame(np.random.randn(30, 5))
    # y = pd.Series(np.random.randn(30))
    # clf = DecisionTree("criterion",max_depth = 5)
    # clf.fit(X,y)
    # yhat = clf.predict(X)
    # print(yhat)

#     df_reg = pd.read_excel('realestate.xlsx')
#     df_reg = df_reg.drop(['No'],axis = 1)
#     df_reg.head()
#     df_reg = df_reg.dropna()
#     attb = list(df_reg.columns)
#     attb.remove('y')
#     train = df_reg[:310]
#     test = df_reg[310:]
#     train_label = train['y']
#     train = train.drop(['y'],axis=1)
#     test_label = test['y']
#     test = test.drop(['y'],axis=1)
#     clf = DecisionTree("criterion",max_depth=5)
#     clf.fit(train,train_label)
#     y = clf.predict(test)
#     print(y)

#The regression tree code is inspired from https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ 
# and https://github.com/sdeepaknarayanan/Machine-Learning/tree/master/Assignment%201