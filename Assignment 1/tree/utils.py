import math 
import pandas as pd

def entropy(Y):

    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """

    if (isinstance(Y,pd.Series)):
        Y = Y.tolist()
    Y_set = set(Y)
    Y_set = list(Y_set)
    a = []
    for i in Y_set:
        a.append(Y.count(i))
    total = sum(a)
    ent = 0
    for i in range (len(Y_set)):
        if a[i]!=0:
            ent = ent - (a[i]/total)*(math.log2(a[i]/total))
    return ent

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    if (isinstance(Y,pd.Series)):
        Y = Y.tolist()
    Y_set = set(Y)
    Y_set = list(Y_set)
    a = []
    Y = list(Y)
    for i in Y_set:
        a.append(Y.count(i))
    total = sum(a)
    gini = 1
    for i in range (len(Y_set)):
        gini = gini - (a[i]/total)**2
    return gini

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """

    initial_gain = entropy(Y)
    Y = Y.tolist()
    # print(Y)
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        initial_gain = initial_gain-(len(l)/len(Y))*entropy(l)
    return initial_gain

def gini_gain(Y, attr):
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    initial_gain=0
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        initial_gain = initial_gain+(len(l)/len(Y))*gini_index(l)
    return initial_gain

def variance_gain(Y, attr):

    initial_var = Y.var()
    Y = Y.tolist()
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        l = pd.Series(l)
        initial_var = initial_var-(len(l)/len(Y))*l.var()
    return initial_var

def inf_gain(Y):
    if (isinstance(Y,pd.Series)):
        Y = Y.tolist()
    Y_set = set(Y)
    Y_set = list(Y_set)
    a = []
    Y = list(Y)
    for i in Y_set:
        a.append(Y.count(i))
    total = sum(a)
    gini = 1
    for i in range (len(Y_set)):
        gini = gini - (a[i]/total)**2
    return gini