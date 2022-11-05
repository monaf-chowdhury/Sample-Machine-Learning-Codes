# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:54:17 2022

@author: monaf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
np.printoptions(suppress = True)


col = ['Exam_1','Exam_2','Pass']
data = pd.read_csv("D:\Coding\Spyder\Sejuti Ma'am\Logistic Regression\data\ex2data1.csv",names=col,header=None)
x = data.iloc[:,:2].values
y = data.iloc[:,-1].values

# Doing z score normalization
m,n = x.shape
for i in range(n):
    x[:,i] = ( x[:,i] - np.mean(x[:,i]) ) / np.std(x[:,i])

# splitting the dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=1)

# Applying sigmoid 
def sigmoid(z):
    temp = 1+ np.exp(-z)
    return 1/temp

# Calculating cost function
def cost_function(x,y,b,w):
    temp = b+np.dot(x,w)
    hyp = sigmoid(temp)
    left = y * np.log(hyp)
    right = (1-y) * np.log(1-hyp)
    err = sum(left+right)
    total_cost = -err/m
    return total_cost

# Calculating Gradient
def gradient(x,y,b,w):
    m,n = x.shape
    temp = b+np.dot(x,w)
    hyp = sigmoid(temp)
    dj_db = np.sum(hyp - y)/m
    dj_dw = np.zeros(n)
    for i in range(n):
        temp = (hyp-y) * x[:,i]
        dj_dw[i] = sum(temp)/m
        
    return dj_db,dj_dw

# Calculating Gradient descent
def Gradient_descent(x,y,alpha = 0.01, iter = 10000):
    m,n = x.shape
    b = 0
    w = np.zeros(n)
    total_cost = []
    for i in range(iter):
        cost = cost_function(x, y, b, w)
        total_cost.append(cost)
        dj_db,dj_dw = gradient(x, y, b, w)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i%1000 ==0:
            print(f'Total cost = {cost}')
    return b,w,total_cost

b,w,total_cost = Gradient_descent(xtrain, ytrain)
plt.plot (total_cost)

# Predicting the value
def predict(x,y,b,w):
    temp = b+ np.dot(x,w)
    hyp = sigmoid(temp)
    ypred = (hyp >0.5) * 1   # this converts the values into 0 or 1
    return ypred

ypred = predict(xtest,ytest,b,w)


################# From sklearn for checking if the answers are correct #################
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=0).fit(xtrain,ytrain)
ycode = reg.predict(xtest)

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(ytest, ycode)
print(cfm)
