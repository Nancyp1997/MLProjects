#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:22:53 2020

@author: nancyscarlet
"""
import pandas as pd
import numpy as np
import matplotlib as plt

dataset=pd.read_csv('/Users/nancyscarlet/Desktop/MLTraining/DataSets/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

dataset.head()
dataset.columns
dataset.size

