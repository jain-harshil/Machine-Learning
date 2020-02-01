import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.datasets import load_iris

np.random.seed(42)

# Read IRIS data set
# ...
# 

def check_accuracy(test,tree):
    temp = 0
    for i in test:
        val = predict(tree,i)
        if i[-1]==val:
            temp = temp+1
    temp /=len(test)
    return temp*100
    
def spl(dataset,classes):
    best_so_far = np.inf
    best_splits = 0
    feature = np.inf
    value_split = 1000
    for i in range(len(dataset[0])-1):
        for elem in dataset:
            new1 = []
            new2 = []
            for el in dataset:
                if el[i]<elem[i]:
                	new1.append(el)
                else:
                    new2.append(el)
            splits = new1,new2
            gscore = gini(splits,classes)
            if gscore<best_so_far:
                best_so_far = gscore
                bs = splits
                f = i
                vs = elem[i]
    return {'split':bs,'Feature':f,'Value':vs}

def gini(splits,classes):
    total_rows = 0
    f = 0
    for i in range(len(splits)):
        total_rows+=len(splits[i])
    f = 0
    for split in splits:
        if len(split)==0:
            continue
        g = 0
        for cls in classes:
            p = [row[-1] for row in split].count(cls)/len(split)
            g = g+p**2
        f+= (1-g)*(len(split)/total_rows)
    return f

def final_node(split):
    cls = [elem[-1] for elem in split]
    count = {}
    for i in cls:
        if i not in count:
            count[i]=1
        else:
            count[i] = count[i]+1
    return max(count, key = count.get)

def predict(node, row):
    f = row[node['Feature']]
    left = node['left']
    right = node['right']
    if  f < node['Value']:
        if isinstance(left, dict):
            return predict(left, row)
        else:
            return left
    else:
        if isinstance(right, dict):
            return predict(right, row)
        else:
            return right


def part(node, maxd, mins, depth):
    
    l, r = node['split']
    if not l or not r:
        node['left']= final_node(l+r)
        node['right'] = final_node(l+r)
        return
    if depth>maxd:
        node['left'] = final_node(l)
        node['right'] = final_node(r)
        return
    if len(l)<=mins:
        node['left'] = final_node(l)
    else:
        node['left'] = spl(l,cls)
        part(node['left'],maxd,mins, depth+1)
    if len(r)<=mins:
        node['right'] = final_node(r)
    else:
        node['right'] = spl(r,cls)
        part(node['right'],maxd,mins,depth+1)

def tree_grow(d, ma, mi):
    root = spl(d,cls)
    part(root,ma,mi,1)
    return root

dataset = load_iris()
X, yhat = dataset.data, dataset.target 
X = pd.DataFrame(X)
y = pd.Series(yhat)
y = y.astype("category")
clf = DecisionTree("gini_index",max_depth = 6)
clf.fit(X[:105],y[0:105])
yhat = clf.predict(X[105:])
X = pd.DataFrame(X)

def nested_cross_validation(dataset):
    for i in range(5):
        test = dataset[30*i:30*i+30]
        if i==0:
            train = dataset[30*i+30:]
        else:
            train1 = dataset[0:30*i]
            train2 = dataset[30*i+30:]
            train = np.append(train1,train2,axis=0)
        maindict = {}
        for depth in range(1,5):
            s = 0
            for j in range(4):
                val = train[30*j:30*j+30]
                train_2 = train[0:30*j]
                train_1 = train[30*j+30:]
                train_aggregate = np.append(train_1,train_2,axis = 0)
                tree = tree_grow(train_aggregate,depth,0)
                acc = check_accuracy(val,tree)
                s = s+acc
            maindict[depth] = s/4
        value = max(maindict, key = maindict.get)
        tree = tree_grow(train,value,0)
        print("Accuracy is,",check_accuracy(test,tree), " for iter",i+1, ". Hence, the optimal depth is",value)


def kfold_cross_validation(dataset,y):
    for i in range(5):
        test = dataset[30*i:30*i+30]
        yhat = y[30*i:30*i+30]
        if i==0:
        	train = dataset[30*i+30:]
        	label = y[30*i+30:]
        else:
        	train1 = dataset[0:30*i]
        	label1 = y[0:30*i]
        	train2 = dataset[30*i+30:]
        	label2 = y[30*i+30:]
        	train = np.append(train1,train2,axis=0)
        	label = label1.append(label2)
        clf = DecisionTree("gini_index",max_depth = 10)
        train = pd.DataFrame(train)
        label = pd.Series(label)
        test = pd.DataFrame(test)

        clf.fit(train,label)	
        yout = clf.predict(test)
        print(accuracy(yhat,yout))

print("The accuracy of 30% test dataset is : ",end = "")
print(accuracy(y[105:],yhat))
print("The precision with respect to class 0 (iris-setosa) is : ",end = "")
print(precision(y[105:],yhat,0))
print("The precision with respect to class 1 (iris-versicolor) is : ",end = "")
print(precision(y[105:],yhat,1))
print("The precision with respect to class 2 (iris-virginica) is : ",end = "")
print(precision(y[105:],yhat,2))

print("The precision with respect to class 0 (iris-setosa) is : ",end = "")
print(recall(y[105:],yhat,0))
print("The precision with respect to class 1 (iris-versicolor) is : ",end = "")
print(recall(y[105:],yhat,1))
print("The precision with respect to class 2 (iris-virginica) is : ",end = "")
print(recall(y[105:],yhat,2))

print("The accuracy for each held out 20% test dataset in 5 fold cross validation is : ")
kfold_cross_validation(X,y)
df = pd.concat([X, y], axis=1, sort=False)
cls = []
dataset = np.array(df)
for i in range(len(dataset)):
	cls.append(dataset[i][-1])
cls = np.unique(np.array(cls))

nested_cross_validation(np.array(df))