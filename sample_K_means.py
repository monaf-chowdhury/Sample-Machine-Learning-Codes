# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:39:28 2022

@author: monaf
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math
np.set_printoptions(suppress = True)

## K means clustering starts here ##
data = pd.read_csv("Data\Mall_Customers.csv")
x = data.iloc[:,3:].values   
plt.scatter(x[:,0],x[:,1])
plt.show() 
k = 5 # letting 5 clusters

# randomly initializing centroids
def determining_centroid(x,k):    
    m,n = x.shape
    centroid = np.zeros((k,n)) # 5X2 size er centroid matrix 
    num = np.random.randint(low=0,high=m,size=k) # random number generator
    for i in range(k):
        val = x [num[i]]
        centroid[i] = val
    return centroid

centroid = determining_centroid(x, k)

# eucleadian distance between centroid and the sample 
def eucleadian_distance(x,y):
    left = (x[0]-y[0])**2
    right = ( x[1] - y[1] ) **2
    return np.sqrt(left+right)

# according to the eucleadin distance closest centroid has been assigned to ith sample 
def assigning_centroid(x,k,centroid):
    m,n = x.shape
    assign_clus = np.zeros(m)
    for i in range(m):
        temp_clus = np.zeros(k)
        for j in range(k):
            temp_clus[j] = eucleadian_distance(x[i],centroid[j]) # here distance from all the clusters are calculated
        assign_clus[i] = np.argmin(temp_clus) # minimum distance cluster is assigned
    return assign_clus
    
# k means for a single iteration
def Kmeans(x,k,centroid):
    m,n = x.shape
    assign_clus = assigning_centroid(x, k, centroid)
        
    # here all the assigned cluster for ith sample are stored in the stored_sample 
    for j in range(k):
        stored_sample = np.zeros((m,n))
        for i in range(m):
            if assign_clus[i] == j:
                stored_sample[i] = x[i]
    
    ## here stored_sample with zero rows are removed for perfect mean calculation
        stored_sample = stored_sample[~(stored_sample == 0).all(axis=1)]
    # mean is calculated along columns 
        centroid[j]  = stored_sample.mean(axis=0)
    return centroid

# iterative k means is done to get perfect centroid
def iterative_k_means(x,k,centroid,iter=10000):
    m,n = x.shape
    for i in range(iter):
        centroid = Kmeans(x,k,centroid)
    return centroid

centroid = iterative_k_means(x, k, centroid)

# Finally, final assignment is done 
assign_clus = assigning_centroid(x, k, centroid)

# here a dictionary stores which cluster has how many variables 
dic={}
for i in range(5):
    dic[i]=sum(assign_clus==i)
print(dic)

##### Plotting cluster #######
plt.scatter(x[assign_clus == 0, 0], x[assign_clus == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[assign_clus == 1, 0], x[assign_clus == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[assign_clus == 2, 0], x[assign_clus == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[assign_clus == 3, 0], x[assign_clus == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[assign_clus == 4, 0], x[assign_clus == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(centroid[:, 0], centroid[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

