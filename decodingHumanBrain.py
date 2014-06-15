'''
Created on May 26, 2014

@author: Aaron
'''

from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np

subjects_train = range(1, 3)   

def loadData(user):
    """Load the data from a file for a specific user"""
    filename = 'data/train_subject%02d.mat' % user
    return loadmat(filename, squeeze_me=True)

def getFeaturesFromDataConsecutiveChannels(data):
     """Get the features from the data. Within each trial, each time unit for every node is a different feature"""
     XX = data['X']
     yy = data['y']
     XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
     XX -= XX.mean(0)
     XX = np.nan_to_num(XX / XX.std(0))
     return (XX, yy)
    
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
    
def getAccuracyForAllModelsForUser(user, models):
    """Get the predictions using the dict of models against one user and print the accuracy """
    print "Training all the models for user " + str(user)
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
        
def main():
    getAccuracyForAllModelsForAllUsers()

main()
    
        
        
    
