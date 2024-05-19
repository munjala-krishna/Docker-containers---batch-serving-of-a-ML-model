# inference.py


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import classification_report


def inference():

    # MODEL_DIR = '/Model_Batch_Service/EEG-letters-main'
    MODEL_PATH_KNC = 'knc.joblib'
    MODEL_PATH_GNB = 'gnb.joblib'
    MODEL_PATH_ABC = 'abc.joblib'
    MODEL_PATH_LDA = 'lda.joblib'
    MODEL_PATH_NN  = 'nn.joblib'
        
    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values
    x_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")
    print(x_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    x_test = preprocessing.normalize(x_test, norm='l2')
    

    # Models training

    # KNeighbors Classification Model
    print(MODEL_PATH_KNC)
    clf_knc = load(MODEL_PATH_KNC)
    print('KNC score and classification:')
    prediction_knc = clf_knc.predict(x_test)
    report_knc = classification_report(y_test, prediction_knc)
    print(clf_knc.score(x_test, y_test))
    print('KNC Prediction:', prediction_knc)
    print('KNC Classification Report:', report_knc )
    print('-'*100)

    # Gaussian Naive Bayes Classification Model  
    print(MODEL_PATH_GNB) 
    clf_gnb = load(MODEL_PATH_GNB)
    print('GNB score and classification:')
    prediction_gnb = clf_gnb.predict(x_test)
    report_gnb = classification_report(y_test, prediction_gnb)
    print(clf_gnb.score(x_test, y_test))
    print('GNB Prediction:', prediction_gnb)
    print('GNB Classification Report:', report_gnb)
    print('-'*100)

    # AdaBoost Classifier Model
    # print(MODEL_PATH_ABC)
    # clf_abc = load(MODEL_PATH_ABC)
    # print('ABC score and classification:')
    # prediction_abc = clf_abc.predict(x_test)
    # report_abc  = classification_report(y_test, prediction_abc)
    # print(clf_abc.score(x_test, y_test))
    # print('ABC Prediction:', prediction_abc)
    # print('ABC Classification Report:', report_abc)
    # print('-'*100)

    # LDA Classification Model
    # print(MODEL_PATH_LDA)
    # clf_lda = load(MODEL_PATH_LDA)
    # print("LDA score and classification:")
    # prediction_lda = clf_lda.predict(x_test)
    # report_lda = classification_report(y_test, prediction_lda)
    # print(clf_lda.score(x_test, y_test))
    # print('LDA Prediction:', prediction_lda)
    # print('LDA Classification Report:', report_lda)
        
    # NN Classification Model
    # clf_nn = load(MODEL_PATH_NN)
    # print("NN score and classification:")
    # prediction_nn = clf_nn.predict(x_test)
    # report_nn = classification_report(y_test, prediction_nn)
    # print(clf_nn.score(x_test, y_test))
    # print('NN Prediction:', prediction_nn)
    # print('NN Classification Report:', report_nn)
    
    
if __name__ == '__main__':
    inference()
