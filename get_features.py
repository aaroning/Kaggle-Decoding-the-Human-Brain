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

def getFeaturesFromDataConsecutiveChannels(data):
     """Get the features from the data. Within each trial, each time unit for every node is a different feature"""
     XX = data['X']
     yy = data['y']
     XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
     XX -= XX.mean(0)
     XX = np.nan_to_num(XX / XX.std(0))
     return (XX, yy)
 
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