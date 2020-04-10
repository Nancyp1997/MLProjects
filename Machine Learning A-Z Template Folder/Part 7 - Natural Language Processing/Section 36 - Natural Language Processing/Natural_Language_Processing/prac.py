#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:23:22 2020

@author: nancyscarlet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating bOW model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

def print_metrics(cm,classif):
    
    accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    precision = (cm[0][0])/(cm[0][0]+cm[0][1])
    recall = (cm[0][0]/(cm[0][0]+cm[1][0]))
    f1score = 2*precision*recall/(precision+recall)
    print(classif,"\nAccuracy:",accuracy,"\nRecall: ",recall, "\nPrecision: ",precision, "\nF1 score: ",f1score)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


#Classifying data

#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
m1 = confusion_matrix(y_test,y_pred)
print_metrics(m1,"Naive Bayes")


#Kernel rbf SVC
from sklearn.svm import SVC
classif = SVC(kernel='rbf',random_state=0)
classif.fit(X_train,y_train)
y_pred = classif.predict(X_test)
m2 = confusion_matrix(y_test,y_pred)
print_metrics(m2,"SVC")


#Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier
clas = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
clas.fit(X_train,y_train)
y_pred = clas.predict(X_test)
m3=confusion_matrix(y_test,y_pred)
print_metrics(m3,"RandomForest")
