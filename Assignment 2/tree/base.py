"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

##### References Used Are ########
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from pprint import pprint
import random

np.random.seed(42)

global criteria_value

###### Real Input and Discrete Ouput  #########

def test_split(index, value, dataset):
  left, right = list(), list()
  for row in dataset:
    if row[index] < value:
      left.append(row)
    else:
      right.append(row)
  return left, right
 
def gini_index(groups, classes):
  n_instances = float(sum([len(group) for group in groups]))
  gini = 0.0
  for group in groups:
    size = float(len(group))
    if size == 0:
      continue
    score = 0.0
    for class_val in classes:
      p = [row[-2] for row in group].count(class_val) / size
      score += p * p
    gini += (1.0 - score) * (size / n_instances)
  return gini
 
def info_gain(groups, classes):
  n_instances = float(sum([len(group) for group in groups]))
  init_gain = float(0)
  group_total = []
  group_total.extend(groups[0])
  group_total.extend(groups[1])
  size = float(len(group_total))
  totscore = float(0)
  if size == 0:
    totscore = 0
  else:
    for j in classes:
      p = 0
      for r in group_total:
        if(r[-2]==j):
          p = p+ r[-1]
      totscore = totscore - p*np.log2(p+10e-9)

  for i in groups:
    size = float(len(i))
    if size == 0:
      continue
    score = 0.0
    for j in classes:
      p = 0
      for r in i:
        if(r[-2]==j):
            p = p+ r[-1]
      score = score - p*np.log2(p+10e-9)
    init_gain += (score) * (size / n_instances)
  init_gain = totscore - init_gain
  return init_gain

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-2] for row in dataset))
    if(criteria_value == "information_gain"):
        b_index, b_value, b_score, b_groups = -1,-1,-1,None
    else:
        b_index, b_value, b_score, b_groups = 10000,10000,10000,None
    
    for index in range(len(dataset[0])-2):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            if(criteria_value == "information_gain"):
                info = info_gain(groups, class_values)
                if info > b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], info, groups
            else:
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
    classes = list(set(row[-2] for row in group))
    max_cls = -2
    max_val = 0
    for class_val in classes:
        p = 0
        for row in group:
            if(row[-2]==class_val):
                p+= row[-1]
        if(p>=max_val):
            max_val = p
            max_cls = class_val
    return max_cls

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
  left, right = node['groups']
  del(node['groups'])
  # check for a no split
  if not left or not right:
    node['left'] = node['right'] = to_terminal(left + right)
    return
  # check for max depth
  if depth >= max_depth:
    node['left'], node['right'] = to_terminal(left), to_terminal(right)
    return
  # process left child
  if len(left) <= min_size:
    node['left'] = to_terminal(left)
  else:
    node['left'] = get_split(left)
    split(node['left'], max_depth, min_size, depth+1)
  # process right child
  if len(right) <= min_size:
    node['right'] = to_terminal(right)
  else:
    node['right'] = get_split(right)
    split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
  root = get_split(train)
  split(root, max_depth, min_size, 1)
  return root
 
# Print a decision tree
def print_tree(node, depth=0):
  if isinstance(node, dict):
    print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
    print_tree(node['left'], depth+1)
    print_tree(node['right'], depth+1)
  else:
    print('%s[%s]' % ((depth*' ', node)))


#### Predict Real Input and Discrete Output
def predict_1(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_1(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_1(node['right'], row)
        else:
            return node['right']

class DecisionTree():
    def __init__(self, criterion, max_depth=100):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to        """
        global criteria_value
        criteria_value = criterion
        self.max_depth = max_depth
        self.bool = 0
        self.num = -1
        self.tree_fit = dict()
        self.label_class = []

    def fit(self, X, y,weights=None):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if(weights is None):
            weights = np.ones(len(y))/len(y)
        self.label_class = y
        if(str(X.dtypes[0])=="float64" and str(y.dtype)=="category" or str(y.dtype)=="object"):
            self.bool = 1
            self.num = 2
            df = pd.concat([X, y, pd.Series(weights)], axis=1, sort=False)
            arr = np.array(df)
            self.tree_fit = build_tree(arr, self.max_depth, 1)
            # self.plot()
            
        return self.tree_fit

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        """
        if(self.num == 2):
            y_pred = []
            for i in np.array(X):
                val = predict_1(self.tree_fit,i)
                y_pred.append(val)
            return pd.Series(y_pred)
        elif(self.num == 3):
            y_pred = np.zeros(len(X))
            for i in range(len(X)):
                val = regressor_predict(self.tree_fit,X.iloc[i])
                y_pred[i] = val
            return pd.Series(y_pred)
        elif(self.num == 1):
            y_pred = []
            arr_1 = np.array(self.label_class)
            for i in range(len(arr_1)):
                val = random.choice(arr_1)
                y_pred.append(val)
            return pd.Series(y_pred)
        else:
            y_pred = []
            arr_1 = np.array(self.label_class)
            for i in range(len(arr_1)):
                val = random.choice(arr_1)
                y_pred.append(val)
            return pd.Series(y_pred)

    def plot(self):
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
        if(self.bool == 0):
            pprint(self.tree_fit)
        else:
            print_tree(self.tree_fit)