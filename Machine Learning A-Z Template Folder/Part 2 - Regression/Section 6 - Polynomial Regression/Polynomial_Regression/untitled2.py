#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:57:03 2020

@author: nancyscarlet
"""

 # Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#X = dataset.iloc[:, 1].values
#The above line of code is commented out because we want the features set to be
# a matrix instead of an array. Instead we specify X like below:
X=dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lineReg = LinearRegression()
lineReg.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures
ploy_reg = PolynomialFeatures(degree=4)
X_poly = ploy_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(ploy_reg.fit_transform(X)),color='blue')
plt.title('Prediction using polynomial regression')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.show()
