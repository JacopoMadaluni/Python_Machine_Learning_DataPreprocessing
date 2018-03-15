# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:23:40 2018

@author: Jaco
"""

#data processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd      #import and model datasets 

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


#take care of missing data

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()  
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])    #prima colonna, france = 1 ecc..

#crea 3 diverse colonne per france, germany and spain, you don't want to have spain(3) > france(1)
oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray()

# do same for y
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#Splitting the dataset into the Training set and the Test set

from sklearn.cross_validation import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 0)


#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)




