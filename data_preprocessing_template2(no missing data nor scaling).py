# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:03:15 2018

@author: Jaco
"""

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


#Splitting the dataset into the Training set and the Test set
from sklearn.cross_validation import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 0)


#feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

"""