'''
Created on May 26, 2014

@author: Aaron
'''

from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from get_features import *;
from feature_reduction import *;

subjects_train = range(1, 17)   
    
def fitLRModel(XX, yy):
    """Fit a linear regression model"""
    clf = LogisticRegression(random_state=0)
    clf.fit(XX, yy)
    return clf
    
def fitLRModelAllUsers():
    """Return a dict of linear regression model trained for all users"""
    models = {}
    for subject in subjects_train:
        data = loadData(subject)
        XX, yy = getFeaturesFromDataConsecutiveChannels(data)
        models[subject] = fitLRModel(XX, yy)
    return models

def fitLRModelJackknife():
    """Return a dict of linear regression model trained for all users"""
    models = {}
    for subject in subjects_train:
        data = loadData(subject)
        XX, yy = getFeaturesFromDataConsecutiveChannelsJackknife(subjects_train, subject)
        models[subject] = fitLRModel(XX, yy)
    return models
    
def getAccuracyForAllModelsForUser(user, models):
    """Get the predictions using the dict of models against one user and print the accuracy """
    print "Training all the models for user " + str(user) + "\n"
    XX, yy = getFeaturesFromDataConsecutiveChannels(loadData(user))
    for key in models:
        y_pred = models[key].predict(XX)
        print "Accuracy score for user " + str(user) + " using model trained with user " + str(key) + " is "
        print accuracy_score(yy, y_pred)
        #print classification_report(yy, y_pred)

def getAccuracyForAllModelsForAllUsers():
    models = fitLRModelAllUsers()
    for subject in subjects_train:
        getAccuracyForAllModelsForUser(subject, models)

def getAccuracyForAllModelsJackknife():
    models = fitLRModelJackknife()
    for subject in subjects_train:
        XX, yy = getFeaturesFromDataConsecutiveChannels(loadData(subject))
        y_pred = models[subject].predict(XX)
        print "Accuracy score for user " + str(subject) + " is "
        print accuracy_score(yy, y_pred)
        
def main():
    XX, yy = get_features_consecutive_channels_all_users(subjects_train)
    categorize_trial(XX, yy, 16)

main()
    
        
        
    
