#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 00:04:38 2020

@author: nancyscarlet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Salary_Data.csv')
dataset.head()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
#test_size = 0.25-> 20% is test and 80% is train

#Importing Linear regressor from scikit learn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train,sample_weight=None)

Y_predicted = regressor.predict(X_test)

#plt.scatter(X_train,Y_train,color='red')
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary based on experience')
plt.xlabel('Experience in Years')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary based on experience')
plt.xlabel('Experience in Years')
plt.ylabel('Salary')
plt.show()