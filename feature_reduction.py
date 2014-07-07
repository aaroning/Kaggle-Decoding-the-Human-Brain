'''
Created on July 6, 2014

@author: Aaron
'''

from sklearn.cluster import KMeans
import numpy as np

def categorize_timeseries(data):
    XX = data['X']
    
    XX = XX.reshape(XX.shape[0] * XX.shape[1], XX.shape[2])
    print XX.shape
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    categorizer = KMeans(n_clusters=5)
    categorizer.fit(XX)
    categories = categorizer.predict(XX)
    print np.sum(categories == 0)
    print np.sum(categories == 1)
    print np.sum(categories == 2)
    print np.sum(categories == 3)
    print np.sum(categories == 4)
    print categories
    
    
def categorize_trial(data, num_clusters):
    XX = data['X']
    yy = data['y']
    
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    categorizer = KMeans(n_clusters=num_clusters)
    categorizer.fit(XX)
    categories = categorizer.predict(XX)
    print "Categories is "
    print categories
    print "yy is "
    print yy 
    categories_range = range(0, 1) 
    for cluster in categories_range:
        print "number of trials in category " + str(cluster)
        print np.sum(categories ==  cluster)
        print "ratio of positives for category " + str(cluster)
        ratio_vector = np.logical_and((categories == cluster), (yy == 1))
        print ratio_vector
        print ratio_vector.sum(0)
        print "categories size "
        print categories.size
        print (ratio_vector.sum(0) / categories.size)