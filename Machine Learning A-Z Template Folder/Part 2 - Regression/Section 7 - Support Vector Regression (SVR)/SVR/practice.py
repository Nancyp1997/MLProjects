#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 23:53:50 2020

@author: nancyscarlet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X1 = dataset.iloc[:, :-1].values
X=dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X=sc_X.fit_transform(X)
y=sc_Y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

y_pred = sc_Y.inverse_transform( regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("TiTLE")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel("position level")
plt.ylabel("Salary")
plt.plot()