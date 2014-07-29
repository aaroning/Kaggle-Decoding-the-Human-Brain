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
import csv;
import re;
from sklearn.decomposition import PCA

subjects_train = range(1, 17)
subjects_test = range(17, 24)   
    
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

def fitLRModelClusters(tmin, tmax):
    """Return a dict of linear regression model trained for all clusters excluding outlier clusters"""
    models = {}
    clusters = [0,1,2]
    
    filename = '3clusters.csv'
    clusters_file = np.genfromtxt(filename, delimiter=',', skip_header=1)[0:9414]
    

    
    XX, yy = get_features_consecutive_channels_all_users(subjects_train, subjects_test, tmin, tmax)
    print "XX shape before PCA" + str(XX.shape)
    print "yy shape before PCA" + str(yy.shape)
    pca = PCA()
    XX = pca.fit_transform(XX)
    print "XX shape " + str(XX.shape)
    print "yy shape " + str(yy.shape)
    XX_train = XX[0:9414]
    print "XX_train shape " + str(XX_train.shape)
    XX_test = XX[9414:13472]
    print "XX_test shape " + str(XX_test.shape)

    
    for cluster in clusters:
        print "Training model for cluster " + str(cluster)
        cluster_XX = XX[clusters_file == cluster]
        cluster_yy = yy[clusters_file == cluster]
        print "XX shape for cluster " + str(cluster) + " equals " + str(cluster_XX.shape)
        print "yy shape for cluster " + str(cluster) + " equals " + str(cluster_yy.shape)
        models[cluster] = fitLRModel(cluster_XX, cluster_yy)
        print "Finished training model for cluster " + str(cluster)
           
    #print "Fitting general model"
    #general_model = fitLRModel(XX, yy)
    #print "Done fitting general model"
    
    ids_result = get_test_ids(subjects_test, tmin, tmax)
    
    print "ids_result shape " + str(ids_result.shape)
    
    test_clusters = np.genfromtxt(filename, delimiter=',', skip_header=1)[9414:13472]
    print "test_clusters shape " + str(test_clusters.shape)
    filename_submission = "submission.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(ids_result.size):
        cluster_number = test_clusters[i]
        #if cluster_number in clusters:
        prediction = models[cluster_number].predict(XX_test[i])
        #else:
        #    prediction = general_model.predict(XX_test[i])
        print >> f, str(ids_result[i]) + "," + str(prediction)
    f.close()
    
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
        
def fixSubmissionFormat():
    f = open('parsed_submission.csv', "w")
    with open('submission.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            print >> f, row[0] + ',' +  row[1].lstrip('[').rstrip(']')
        
        
def main():
    tmin = 0
    tmax = 1
#    XX, yy = get_features_consecutive_channels_all_users(subjects_train, subjects_test, tmin, tmax)
 #   categorize_trial(XX, yy, 3);
#    categorize_trial(XX, yy, 16)
#    categorize_trial(XX, yy, 23)
    #categorize_trial(XX, yy, 40)
#    models = fitLRModelClusters()
    fitLRModelClusters(tmin, tmax)

    fixSubmissionFormat()
main()
    
        
        
    
