#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:21:10 2020

@author: maylamm
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

from sklearn import metrics

from sklearn.model_selection import train_test_split


# Read Data 
data = pd.read_csv('spambase.data', header=None)

X = data.iloc[:,0:57]

y = data.iloc[:,57]


# Split Data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# Decision Tree 
dt = tree.DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)

# Plot decision tree 
fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(dt, filled=True)

fig.savefig("decision_tree.png")






# Random Forest 
n_trees = list(range(1,51))
rf_rates = []

for n in range(1,51): 
    rf = RandomForestClassifier(n_estimators=n).fit(X_train, y_train)

    rf_pred=rf.predict(X_test)

    rf_rate = 1-metrics.accuracy_score(y_test, rf_pred)
    
    rf_rates.append(rf_rate)
    
# Plot misclassification rate for random forest 
fig = plt.figure(figsize=(25,20))
plt.plot(n_trees, rf_rates)
plt.title('Misclassification Rate for Random Forest', fontsize=40)
plt.ylabel('misclassification rate', fontsize=40)
plt.xlabel('#trees', fontsize=40)
fig.savefig("rf_rates.png")


# Plot misclassification rate for decision tree
dt_pred = dt.predict(X_test)
dt_rate = 1-metrics.accuracy_score(y_test, dt_pred)
dt_rates = [dt_rate]*50
fig = plt.figure(figsize=(25,20))
plt.plot(n_trees, dt_rates)
plt.title('Misclassification Rate for Decision Tree', fontsize=40)
plt.ylabel('misclassification rate', fontsize=40)
plt.xlabel('#trees', fontsize=40)
fig.savefig("dt_rates.png")


# extract non-spam(0) emails
X_nonspam = X_train[y_train==0]


# SVM for novelty (spam, -1) detection 
svm = OneClassSVM(gamma='scale').fit(X_nonspam)
svm_pred = svm.predict(X_test)
for i in range(0, len(svm_pred)):
    if svm_pred[i]==1:
        svm_pred[i] = 0
    if svm_pred[i]==-1:
        svm_pred[i] = 1
     
svm_rate = 1-metrics.accuracy_score(y_test, svm_pred)

    


