#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:19:54 2020

@author: nancyscarlet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
 
from sklearn.cluster import KMeans
 
 #Finding the optimised num of clusters by elbow
 #graph plotting
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',
                     n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    #Inertia gives the WCSS value for a particular i
     
plt.plot(range(1,11),wcss)
plt.title('Elbow graph')
plt.xlabel('Num of neighbors')
plt.ylabel('WCSS')
plt.show()

#FRom the graph we saw that the elbow is at k=5
#k is the num of neighbors
kmeans=KMeans(n_clusters=5,init='k-means++',
                     n_init=10,max_iter=300,random_state=0)
y_kmeans= kmeans.fit_predict(X)

#X[y_kmeans==0,0] implies selecting that row only if 
#y is 0
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='green',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='blue',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',labek='Centroids')
plt.title('k means clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()