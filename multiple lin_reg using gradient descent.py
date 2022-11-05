# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 05:42:01 2022

@author: monaf
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
data = pd.read_csv("Data\Real estate.csv")
x = data.iloc[:,1:7].values # help 
y = data.iloc[:,-1].values
m,n = x.shape

# Z score normalization
for i in range(n):
    x[:,i] = ( x[:,i] - np.mean(x[:,i]) )/ np.std(x[:,i])

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.75,random_state=1)

# calculating cost function 
def cost_function(x,y,b,w):
    temp = b+np.dot(x,w)
    temp = (temp - y) **2 
    total_cost = sum(temp) / (2*m)
    return total_cost

# calculating gradient 
def gradient(x,y,b,w):
    m,n = x.shape
    hypothesis = b+ np.dot(x,w)
    dj_db = sum(hypothesis - y) / m
    dj_dw = np.zeros(n)
    for i in range(n):
        temp = hypothesis - y 
        val = temp * x[:,i]
        dj_dw[i] = sum(val)/m
        
    return dj_db,dj_dw
# Doing gradient descent
def Gradient_descent(x,y,iter = 1000,alpha = 0.1):
    m,n = x.shape
    b = 0
    w = np.zeros(n)
    total_cost = []
    
    for i in range(iter):
        cost = cost_function(x, y, b, w)
        total_cost.append(cost)
        dj_db,dj_dw = gradient(x,y,b,w)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        
        if i%50==0:
            print(f'The total cost: {cost}')
    return b,w 

b,w = Gradient_descent(xtrain, ytrain,1000,0.1)

def predict(x,y,b,w):
    hypo = b+np.dot(x,w)
    return hypo 

result = np.array(predict(xtest,ytest,b,w))

####### Took the following plotting code from github ########

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xtrain,ytrain)
# Predicting Model
ypred = reg.predict(xtest)

from sklearn.metrics import mean_squared_error
print("MSE on train {:.3f}".format(mean_squared_error(ytest,ypred)))
