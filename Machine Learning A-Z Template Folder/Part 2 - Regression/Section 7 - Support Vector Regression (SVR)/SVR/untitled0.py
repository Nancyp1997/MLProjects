#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:14:17 2020

@author: nancyscarlet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

#Predict the salary of the new employee to be hired
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values.reshape(-1,1)

#Not splitting train test because only 10 rows aer there
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X=sc_X.fit_transform(X)
y=sc_Y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

y_pred = regressor.predict(np.array(6.5).reshape(-1,1))
y_sal = sc_Y.inverse_transform(y_pred)

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.show()