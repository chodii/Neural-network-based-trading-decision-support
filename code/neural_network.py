# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:33:24 2022

@author: Jan Chodora

Neural network training and splitting data to be fed into it.
"""

#import keras
#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
#import keras.layers as layers# Neural network
import numpy as np
from tensorflow import autograph as autogr_decor
import logging

import market

#from sklearn.metrics import accuracy_score

def split_data_set(X, y):
    """
    Splits data in ration 80%/20%, 80% for train set, 20% for test set
    :return: X_train, y_train, X_test, y_test
    """
    rng = int(len(y)*0.8)
    # dataset splitting and formating
    X_train = []
    X_test = []
    for i in range(len(X)):
        X_train.append(X[i][:rng])
        X_test.append(X[i][rng:])
    
    y_train = y[:rng]
    y_test = y[rng:]
    return X_train, y_train, X_test, y_test


@autogr_decor.experimental.do_not_convert
def feed_NN(X_train, y_train, X_test, epochs=120, batch_size=5, hidden_neurons = 8):
    """
    :param X_train: training data
    :param y_train: expected output for training data
    :param X_test: test data for which :y_pred: is generated
    :param epochs: how many epochs will the model be trained on
    :param batch_size: size of batch, among with :epochs: used for training
    :hidden_neurons: number of neurons in the hidden layer of the model.
        Number of neurons in input(X) and output(y) layer is set by character of the data.
    :return: predicted output for vector X_test
    """
    autogr_decor.set_verbosity(0)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)#supress warnings
    
    y_train = np.array(y_train)
    # y_test = np.array(y_test)
    
    X_train = np.transpose(np.array(X_train))
    X_test = np.transpose(np.array(X_test))
    
    
    # NN
    model = Sequential()
    model.add(Dense(X_train.shape[1], activation="relu"))
    #model.add(Dense(8, input_shape=(10, ), activation="relu"))
    model.add(Dense(hidden_neurons, activation="relu"))# 8
    #model.add(Dense(5, activation="relu"))# 8
    model.add(Dense(y_train.shape[1], activation="sigmoid"))# 6# softmax, sigmoid
    
    
    # loss func & optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#sparse_categorical_crossentropy
    
    # training model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)#, verbose=0
    # batch_size - less -> faster & less_accurate
    # test
    y_pred = model.predict(X_test)
    
    return y_pred, history.history["accuracy"], history.history["loss"]



#pip install numpy --upgrade