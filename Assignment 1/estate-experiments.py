
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read real-estate data set
# ...
# 


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

# ScikitLearn Implementation
from sklearn.metrics import mean_absolute_error
from sklearn import tree

df_reg = pd.read_excel('realestate.xlsx')
df_reg = df_reg.drop(['No'],axis = 1)
df_reg.head()
df_reg = df_reg.dropna()
attb = list(df_reg.columns)
attb.remove('y')
train = df_reg[:310]
test = df_reg[310:]
train_label = train['y']
train = train.drop(['y'],axis=1)
test_label = test['y']
test = test.drop(['y'],axis=1)

clf = DecisionTree("criterion",max_depth=4)
clf.fit(train,train_label)
y_pred = np.zeros(len(test))
for i in range (len(test)):
	val = pred_reg(clf.tree,test.iloc[i])
	y_pred[i] = val
y_pred = pd.Series(y_pred)

print("The RMSE with respect to the classifier made is : ",end = "")
print(rmse(y_pred,test_label))
print("The MAE with respect to the classifier made is : ",end = "")
print(mae(y_pred,test_label))

## Scikit Learn Implementation

reg =  tree.DecisionTreeRegressor()
reg.fit(train,train_label)
pred = reg.predict(test)
print("MAE and RMSE with the Scikit Learn Implementation are ",mean_absolute_error(pred, test_label), np.std(np.abs(pred-test_label)))
