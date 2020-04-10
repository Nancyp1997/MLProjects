#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:04:42 2020

@author: nancyscarlet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('emails.csv')

#Dropping duplicates
dataset.drop_duplicates(inplace=True)

#nltk.download('stopwords')

bulk = []
def process_text(tex):
    email = tex
    email = re.sub('[^a-zA-Z]',' ',email)
    email = email.lower()
    email = email.split()
    ps = PorterStemmer()
    email = [ps.stem(word) for word in email if not word in set(stopwords.words('english'))] 
    email = ' '.join(email)
    return email
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
