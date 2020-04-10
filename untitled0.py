#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:36:26 2020

@author: nancyscarlet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('emails.csv')
dataset.drop_duplicates(inplace=True)

def process_text(text):
    email=text
    #Remove punctuation and allow only alphabetical data
    email = re.sub('[^a-zA-Z]',' ',email)
    #Convert to lower case
    email = email.lower()
    print(type(email))
    #Split to words 
    email_words = email.split()
    #Considering only stems
    ps = PorterStemmer()
    email_words = [ps.stem(word) for word in email_words if not word in set(stopwords.words('english'))]
    email = ' '.join(email_words)
    return email
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
unknownarr=[]
#Readying bag of words model
bagOfWords = CountVectorizer(analyzer=process_text).fit_transform(dataset['text'])

#Divide to train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(bagOfWords,dataset['spam'],test_size=0.2,random_state=0)

#Naive Bayes Classifier 
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

print(classifier.predict(X_train))
print(y_train.values)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train,pred))

y_pred_test = classifier.predict(X_test)
print(classification_report(y_test,y_pred_test))
print("Accuracy",accuracy_score(y_test,y_pred_test))

