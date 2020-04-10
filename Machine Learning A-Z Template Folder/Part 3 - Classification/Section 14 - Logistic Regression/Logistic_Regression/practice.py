#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 00:16:19 2020

@author: nancyscarlet
"""

#DATA EXPLORATORY ANALYSIS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset.columns)

#Predict whether user will buy the product or not 
X=dataset.iloc[:,[2,4]].values
Y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
 
#Predictin y_test values
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
accuracyLogisticRegression = (cm[0][0]+cm[1][1])/(cm[1][0]+cm[1][1]+cm[0][0]+cm[0][1])


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)


from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y_test,y_pred)
r2score = r2_score(Y_test,y_pred)
