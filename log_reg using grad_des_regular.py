# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 17:56:14 2022

@author: monaf
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
np.set_printoptions(suppress=True)

col = ['Exam 1','Exam 2','Pass']

# Importing dataset as text 
data = np.loadtxt("Data\ex2data1.txt",delimiter=',')
x = data[:,:2]
y = data[:,-1]


# Mean normalization 
m,n = x.shape 
for i in range(n):
    span = x[:,i].max() - x[:,i].min()
    x[:,i] = ( x[:,i] - np.mean(x[:,i]) ) / span

# Splitting data set
from sklearn.model_selection import train_test_split
xtrain, xtest ,ytrain, ytest = train_test_split(x,y,test_size=0.25,random_state=1)

# sigmoid function
def sigmoid(z):
    temp = 1+ np.exp(-z)
    return 1/temp 

# cost function with regularization
def cost_function(x,y,b,w, lambda_ = 0):
    temp = b+np.dot(x,w)
    hyp = sigmoid(temp) 
    left = y * np.log(hyp)
    right = ( 1- y) * np.log(1-hyp)
    total_error = -np.sum(left+right)/m ## This is the total error without regularization
    # regularizatipn cost 
    temp_val = lambda_/(2*m)
    reg_err = np.sum(w**2) * temp_val
    total_error += reg_err    
    
    return total_error

#Finding gradient with regularization
def gradient(x,y,b,w,lambda_ = 0):
    m,n = x.shape
    temp = b + np.dot(x,w)
    hyp = sigmoid(temp) - y 
    dj_db = np.sum(hyp)/m
    dj_dw = np.zeros(n)
    for i in range(n):
        temp = hyp * x[:,i]
        dj_dw[i] = np.sum(temp)/m
        temp_val = (lambda_/m)*w[i]
        dj_dw[i]+= temp_val
    return dj_db, dj_dw

# Calculating Gradient Descent
def gradient_descent(x,y,alpha=0.1,lambda_=0,iter = 10000):
    m,n = x.shape
    total_cost = []
    b = 0 
    w = np.zeros(n)
    total_cost = []
    for i in range(iter):
        cost = cost_function(x, y, b, w,lambda_)
        total_cost.append(cost)
        dj_db,dj_dw = gradient(x, y, b, w,lambda_)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i%1000 == 0:
            print(f"Total cost = {cost}")
    return b,w,total_cost

b,w,total_cost = gradient_descent(xtrain, ytrain,lambda_= 0.5)
plt.plot(total_cost)

# Predicting dataset
def predict (x,y,b,w):
    temp = b + np.dot(x,w)
    hyp = sigmoid(temp)
    ypred = (hyp>0.5) * 1
    return ypred 

ypred = predict(xtest,ytest,b,w) # predicting the actual value

# confusion matrix
from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(ytest, ypred)
print(cfm)

# checking if the answer is correct
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression().fit(xtrain,ytrain)
ycode = reg.predict(xtest)    

check_cfm = confusion_matrix(ytest, ycode)
print(check_cfm)





################################## New features ##########################

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

# if the dataset is not divideable by a line then new features must be created









