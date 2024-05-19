# tain.py


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing

def train():

    # Load directory paths for persisting model

    # MODEL_DIR = '/Model_Batch_Service/EEG-letters-main'
    MODEL_PATH_KNC = 'knc.joblib'
    MODEL_PATH_GNB = 'gnb.joblib'
    MODEL_PATH_ABC = 'abc.joblib'
    MODEL_PATH_LDA = 'lda.joblib'
    MODEL_PATH_NN = 'nn.joblib'
      
    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    x_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("Shape of the training data")
    print(x_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    x_train = preprocessing.normalize(x_train, norm='l2')
    
    # Models training

    # KNeighborsClassifier (Default Parameters)
    clf_knc = KNeighborsClassifier()
    clf_knc.fit(x_train, y_train)
    # Save model
    from joblib import dump
    dump(clf_knc, MODEL_PATH_KNC)

    # Gaussian Naive Bayes Classifier (Default Parameters)
    clf_gnb = GaussianNB()
    clf_gnb.fit(x_train, y_train)
    # Save model
    from joblib import dump
    dump(clf_gnb, MODEL_PATH_GNB)

    # AdaBoost Classifier (Default Parameters)
    # clf_abc = AdaBoostClassifier()
    # clf_abc.fit(x_train, y_train)
    # Save model
    # from joblib import dump
    # dump(clf_abc, MODEL_PATH_ABC)
    
    # Linear Discrimant Analysis (Default parameters)
    # clf_lda = LinearDiscriminantAnalysis()
    # clf_lda.fit(x_train, y_train)
    # Save model
    # from joblib import dump
    # dump(clf_lda, MODEL_PATH_LDA)
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    # clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    # clf_NN.fit(x_train, y_train)
    # Save model
    # from joblib import dump, load
    # dump(clf_NN, MODEL_PATH_NN)
        
if __name__ == '__main__':
    train()
