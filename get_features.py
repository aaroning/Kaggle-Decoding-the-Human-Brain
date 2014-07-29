'''
Created on June 18, 2014

@author: Aaron
'''
from scipy.io import loadmat
import numpy as np

def loadData(user):
    """Load the data from a file for a specific user"""
    filename = 'data/train_subject%02d.mat' % user
    return loadmat(filename, squeeze_me=True)

def loadTestData(user):
    """Load the data from a file for a specific test user"""
    filename = 'data/test_subject%02d.mat' % user
    return loadmat(filename, squeeze_me=True)

"""def loadTestSet(subjects_test):
    ids_test = []
    
    for subject in subjects_test:
        data = loadTestData(user)
        XX = data['X']
        ids = data['Id']

        XX = create_features(XX, tmin, tmax, sfreq)

        X_test.append(XX)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print "Testset:", X_test.shape"""

def getFeaturesFromDataConsecutiveChannels(data, tmin, tmax):
     """Get the features from the data. Within each trial, each time unit for every node is a different feature"""
     XX = data['X']
     yy = data['y']
     time = np.linspace(-0.5, 1.0, 375)
     time_window = np.logical_and(time >= tmin, time <= tmax)
     XX = XX[:,:,time_window]
     XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
     XX -= XX.mean(0)
     XX = np.nan_to_num(XX / XX.std(0))
     return (XX, yy)
 
def getTestFeaturesFromData(data, tmin, tmax):
     """Get the features from the test data. Within each trial, each time unit for every node is a different feature"""
     XX = data['X']
     time = np.linspace(-0.5, 1.0, 375)
     time_window = np.logical_and(time >= tmin, time <= tmax)
     XX = XX[:,:,time_window]
     ids = data['Id']
     XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
     XX -= XX.mean(0)
     XX = np.nan_to_num(XX / XX.std(0))
     return ids, XX
 
def getFeaturesFromDataConsecutiveChannelsJackknife(subjects_train, jacknifed_subject):
    """Get the features from the data. Within each trial, each time unit for every node is a different feature.
    Append all features together for all users but one"""
    XX_result = None
    yy_result = None
    for subject in subjects_train:
        if subject != jacknifed_subject:
            data = loadData(subject)
            XX, yy = getFeaturesFromDataConsecutiveChannels(data)
            if XX_result is None:
                XX_result = XX
            else:
                XX_result = np.concatenate((XX_result, XX))
            if yy_result is None:
                yy_result = yy
            else:
                yy_result = np.concatenate((yy_result, yy))
    return (XX_result, yy_result)

def get_features_consecutive_channels_all_users(subjects_train, subjects_test, tmin, tmax):
    """Get the features from the data. Within each trial, each time unit for every node is a different feature.
    Append all features together for all users"""
    XX_result = None
    yy_result = None
    for subject in subjects_train:
        print "Loading data for training subject " + str(subject)
        data = loadData(subject)
        XX, yy = getFeaturesFromDataConsecutiveChannels(data, tmin, tmax)
        if XX_result is None:
            XX_result = XX
        else:
            XX_result = np.concatenate((XX_result, XX))
        if yy_result is None:
            yy_result = yy
        else:
            yy_result = np.concatenate((yy_result, yy))
    for subject in subjects_test:
        print "Loading data for test subject " + str(subject)
        data = loadTestData(subject)
        ids, XX = getTestFeaturesFromData(data, tmin, tmax)
        XX_result = np.concatenate((XX_result, XX))
    return (XX_result, yy_result)

def get_test_features_all_users(subjects_test, tmin, tmax):
    """Get the features from the data. Within each trial, each time unit for every node is a different feature.
    Append all features together for all users"""
    XX_result = None
    ids_result = None
    for subject in subjects_test:
        data = loadTestData(subject)
        ids, XX = getTestFeaturesFromData(data, tmin, tmax)
        if XX_result is None:
            XX_result = XX
        else:
            XX_result = np.concatenate((XX_result, XX))
        if ids_result is None:
            ids_result = ids
        else:
            ids_result = np.concatenate((ids_result, ids))
    return (XX_result, ids_result)

def get_test_ids(subjects_test, tmin, tmax):
    ids_result = None
    for subject in subjects_test:
        data = loadTestData(subject)
        ids = data['Id']
        if ids_result is None:
            ids_result = ids
        else:
            ids_result = np.concatenate((ids_result, ids))
    return ids_result
    