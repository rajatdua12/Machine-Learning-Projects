#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:14:44 2019

@author: rajatdua
"""

"""
In this problem, I have analyzed some tweet data for Tesla.  The main goal is to make any sense of it. 
I went through my analyses using following steps: 
•	I have cleaned the data.   
•	I have deleted a lot of garbage and duplicate tweets.  
•	I have labeled the tweets with positive (1) or negative (0) sentiment by hand.   

After that, I have created a convolutional neural network, with embedding and drop out.   
I have used the GloVe embedding weights to see if improves my results.  
That is, I have separated the data into testing and training data and used some of the Tesla tweets 
data set as the test set.
"""

#Import Libraries

from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
#conda install -c anaconda pydot
#conda install -c anaconda graphviz

# Read the Tesla tweet data

df = pd.read_csv( 'TSLA_tweets.csv', header = None, encoding = 'unicode_escape')
tweets = df[0].tolist()

# Clean the data of punctuations, stop words, and short words

tweets_clean = np.empty( len( tweets ), dtype = object )
for i in range( len( tweets ) ):
    tokens = tweets[ i ].split()
    
    # Get rid of punctuations
    
    table = str.maketrans('', '', punctuation)
    
    tokens = [ w.translate( table ) for w in tokens ]
    tokens = [ word for word in tokens if word.isalpha() ]
    
    # Get rid of stop words
    
    stop_words = set( stopwords.words( 'english' ) )
    tokens = [ w for w in tokens if not w in stop_words ]
    
    # Get rid of short words
    
    tokens = [ word for word in tokens if len( word ) > 1 ]
    tokens = ' '.join(tokens)
    tweets_clean[ i ] = tokens[:]
    
# Tokenize and encode the data

tokenizer = Tokenizer()
tokenizer.fit_on_texts( tweets_clean )
max_length = max( [ len( s.split() ) for s in tweets_clean ] )
vocab_size = len( tokenizer.word_index ) + 1
encoded = tokenizer.texts_to_sequences( tweets_clean )

# Pad the encoded sequences to get X

X = pad_sequences( encoded, maxlen = max_length, padding = 'post' )
y = df[ 1 ].values

#############################################################

# Load the GloVe embeddings

embeddings = dict()
GloVe_embeddings = open( '/Users/rajatdua/Documents/Illinois Institute of Technology/Fall 2019 Semester/Databases and Machine Learning/glove.6B.100d.txt', errors = 'ignore' )
for line in GloVe_embeddings:
    values = line.split()
    word = values[ 0 ]
    try:
        weights = np.asarray( values[ 1: ], dtype = 'float32' )
    except ValueError:
        continue
    embeddings[ word ] = weights
GloVe_embeddings.close()
##############################################################

# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 7 )

# Build the convolutional neural network

# Channel 1

inputs1 = Input( shape = ( 17, ) )
embedding1 = Embedding( vocab_size, 100 )( inputs1 )
conv1 = Conv1D( filters = 32, kernel_size = 4, activation = 'relu' )( embedding1 )
drop1 = Dropout( 0.5 )( conv1 )
pool1 = MaxPooling1D( pool_size = 2 )( drop1 )
flat1 = Flatten()( pool1 )

# Channel 2

inputs2 = Input( shape = ( 17, ) )
embedding2 = Embedding( vocab_size, 100 )( inputs2 )
conv2 = Conv1D( filters = 32, kernel_size = 6, activation = 'relu' )( embedding2 )
drop2 = Dropout( 0.5 )( conv2 )
pool2 = MaxPooling1D( pool_size = 2 )( drop2 )
flat2 = Flatten()( pool2 )

# Channel 3

inputs3 = Input( shape = ( 17, ) )
embedding3 = Embedding( vocab_size, 100 )( inputs3 )
conv3 = Conv1D( filters=32, kernel_size = 8, activation = 'relu' )( embedding3 )
drop3 = Dropout( 0.5 )( conv3 )
pool3 = MaxPooling1D( pool_size = 2 )( drop3 )
flat3 = Flatten()( pool3 )

# Combine the channels

combined = concatenate( [ flat1, flat2, flat3 ] )

# Create the dense layers

dense1 = Dense( 10, activation = 'relu' )( combined )
outputs = Dense( 1, activation = 'sigmoid' )( dense1 )
model = Model( inputs = [ inputs1, inputs2, inputs3 ], outputs = outputs )

# Compile and fit the CNN

model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = [ 'accuracy' ] )

#plot_model( model, show_shapes = True, to_file = 'multichannel.png' )

model.fit( [ X_train, X_train, X_train ], y_train, epochs = 10, batch_size = 16 )

# Evaluate the model on training and test data sets

loss, acc = model.evaluate( [ X_train, X_train, X_train ], y_train, verbose = 0 )
print( 'TRAINING ACCURACY: %.2f' % acc )

loss, acc = model.evaluate( [ X_test, X_test, X_test ], y_test, verbose = 0 )
print( '\nTEST ACCURACY: %.2f' % acc )