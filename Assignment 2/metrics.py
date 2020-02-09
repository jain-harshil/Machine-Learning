import pandas as pd
import numpy as np
import math

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    sum1 = 0
    for i in range (len(y_hat)):
        if y_hat[i]==y[i]:
            sum1+=1

    return sum1/len(y_hat)*100
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size), "The size of y_hat and y is not equal"
    # TODO: Write here

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    y_hat_set = list(set(y_hat))
    y_set = list(set(y))
    total_num = max(len(y_hat_set),len(y_set))
    den = y_hat.count(cls)
    a = []
    b = []
    for i in range(len(y_hat)):
        if y_hat[i]==cls:
            a.append(1)
        else:
            a.append(0)
    for i in range (len(y)):
        if y[i]==cls:
            b.append(1)
        else:
            b.append(0)  
    num = int(np.sum(np.array(a)*np.array(b)))
    if den == 0:
        return 1
    return num/den

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    y_hat_set = list(set(y_hat))
    y_set = list(set(y))
    total_num = max(len(y_hat_set),len(y_set))
    den = y.count(cls)
    a = []
    b = []
    for i in range(len(y_hat)):
        if y_hat[i]==cls:
            a.append(1)
        else:
            a.append(0)
    for i in range (len(y)):
        if y[i]==cls:
            b.append(1)
        else:
            b.append(0)  
    num = int(np.sum(np.array(a)*np.array(b)))
    if den == 0:
        return 1
    return num/den

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=y_hat[i]-y[i]
    mae = 0
    for i in range (len(diff)):
        mae+=diff[i]**2
    mae = mae/len(y)
    return math.sqrt(mae)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=abs(y_hat[i]-y[i])
    return sum(diff)/len(y)
