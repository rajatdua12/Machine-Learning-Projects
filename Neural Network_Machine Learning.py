#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:55:04 2019

@author: rajatdua
"""

"""
In this problem, I am architecting a neural network that uses the data to predict the 
category of the following year’s log return on the S&P 500 where the categories are down 
more than one standard deviation, between minus one and plus one standard deviation, and 
up more than one standard deviation.
"""

#Import Libraries

import numpy as np
import pandas as pd
import random
#import talib
import pandas_datareader as web
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None # default='warn'

random.seed( 7 )

#Read the Data from Yahoo

df = web.DataReader('SPY', 'yahoo', start='2014-01-01', end='2019-11-12')

df.head()

#Define the dataset columns

dataset = df[ ['Open', 'High', 'Low', 'Close', 'Adj Close'] ]

#Create columns based on numeric manipulation of existing data

dataset[ 'H-L' ] = dataset[ 'High' ] - dataset[ 'Low' ]
dataset[ 'C-O' ] = dataset[ 'Close' ] - dataset[ 'Open' ]

#Create Columns based on Moving Averages of Adjusted Close Data Column

dataset[ '3day MA' ] = dataset[ 'Adj Close' ].shift( 1 ).rolling( window = 3 ).mean()
dataset[ '10day MA' ] = dataset[ 'Adj Close' ].shift( 1 ).rolling( window = 10 ).mean()
dataset[ '30day MA' ] = dataset[ 'Adj Close' ].shift( 1 ).rolling( window = 30 ).mean()

#Create a column for standard deviation based on 5 days moving average

dataset[ 'Std_dev' ] = dataset[ 'Adj Close' ].rolling( 5 ).std()
#dataset[ 'RSI' ] = talib.RSI( dataset[ 'Close' ].values, timeperiod = 9 )
#dataset[ 'Williams %R' ] = talib.WILLR( dataset[ 'High' ].values, dataset[ 'Low' ].values, dataset[ 'Close' ].values, 7 )

#Create columns based on log return of adjusted close, standard deviation of log return, 
#and mean of log return column

dataset[ 'log_return' ] = np.log(dataset['Adj Close']/dataset['Adj Close'].shift(1))
dataset['Std'] = dataset[ 'log_return' ].std()
dataset['mean'] = dataset[ 'log_return' ].mean()

#Create a column category which matches the conditions mentioned in the question

dataset['category'] = np.where( dataset[ 'log_return' ] > (dataset['mean'] + dataset['Std']), 1, 
       np.where( dataset[ 'log_return' ] < (dataset['mean'] - dataset['Std']) , -1, 0))
print( dataset[ 'category' ])

# Drop 30 rows due to Moving Average calculations

dataset = dataset.dropna() 

#Define Dependent (X) and Independent (y) Variables

S = dataset.values
X = S[ :, 0:-3 ].astype( float )
#print(X)
y = S[ :, -1 ]
#print(y)

#Stansardize and transform the data

scaler = StandardScaler()
X = scaler.fit_transform( X )

#Test-train split the data

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 7 )

#Implement neural network model in the training dataset
 
model = Sequential()
model.add( Dense( units = 30, kernel_initializer = 'random_uniform', bias_initializer = 'zeros', activation = 'sigmoid', input_dim = X.shape[ 1 ] ) )
model.add( Dense( units = 20, activation = 'linear' ) )
model.add( Dense( units = 1, activation = 'sigmoid' ) )

model.compile( optimizer = 'adam', loss = 'mean_squared_error', metrics = [ 'accuracy' ] )
model.fit( X_train, y_train, epochs = 1000, verbose = False )

#Make predictions based on model

predictions = model.predict( X_test )
print( '\nRAW PREDICTIONS:\n' )

print( predictions[ :5 ] )

# Make predictions 0 or 1

predictions = np.round( predictions ) 

#Print categorical predictions

print( '\n CATEGORICAL PREDICTIONS:\n' )
print( predictions[ :5 ] )

#Print Confusion Matrix based on preditions from model

print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test, predictions ) )