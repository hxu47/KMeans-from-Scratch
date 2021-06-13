import random 
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def select_centroids(X, k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly.
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    n, p = X.shape

    # Pick first point randomly
    ind = random.sample(range(n), 1)  
    centroids = X[ind]

    while len(centroids) < k:
        min_distances = []
        for x in X:
            # for each point, compute distance to each cluster
            distances = np.sum((x - centroids)**2, axis=1)  

            # find the minimum distance of this point to one of clusters
            min_distance = min(distances)
            min_distances.append(min_distance)

        # Find the max distance. The associated point is the new centroid.
        point_ind = np.argmax(min_distances)
        new_centroid = np.expand_dims(X[point_ind], axis=0) # same dimension as centroids

        # append the new centroid to centroids
        centroids = np.concatenate((centroids, new_centroid))
    return centroids
    
    
    
def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):
    
    n, p = X.shape
    
    # set up intial centroids
    if centroids is None: 
        # kmeans: select K unique points from X as intial centroids
        ind = random.sample(range(n), k)  # pick k indexes without replacement 
        centroids = X[ind]
    elif centroids == 'kmeans++': 
        # kmeans++: randomly pick the first of k centroids. 
        # Then, pick next k-1 points by selecting points that maximize 
        # the minimum distance to all existing cluster centroids
        centroids = select_centroids(X, k)
    
    labels = np.array([-1]*n, dtype=int)
    n_iter = 0
    while n_iter < max_iter:
        prev_centroids = centroids.copy()
        centroids = []  # initialize current centroid list
        # assign each observation in X to a cluster
        for i, x in enumerate(X):
            # find closest centroid to x
            distances = np.sum((x - prev_centroids)**2, axis=1)  
            label_x = np.argmin(distances)
            # assign x to cluster
            labels[i] = label_x 
        
        # recompute centroids
        for i in range(k):
            cluster_i = X[labels == i]
            centroids.append(np.mean(cluster_i, axis=0))
            
        #  stop if the average norm of centroids-previous_centroids is less than the tolerance
        centroids = np.array(centroids)
        diff = np.linalg.norm(centroids-prev_centroids)
        if diff < tolerance:
            break
            
        n_iter += 1
                
    return centroids, labels



def likely_confusion_matrix(y, labels):
    if sum(y == (1-labels)) > sum(y == labels):
        labels = 1-labels
        confmat = confusion_matrix(y, labels)
    else:
        confmat = confusion_matrix(y, labels)
    print('act\pred | class-0  class-1')
    print('---------------------------')
    print('class-0  |{0:8d} {1:8d}'.format(confmat[0, 0], confmat[0, 1]))
    print('class-1  |{0:8d} {1:8d}'.format(confmat[1, 0], confmat[1, 1]))
    
    print()
    print('Clustering accuracy: ', metrics.accuracy_score(y, labels))
    
    
def elbow_method(X, K):
    # K: a list of number of clusters
    inertia = []
    for k in K:
        # Building and fitting the model
        centroids, labels = kmeans(X, k=k, max_iter=1000, tolerance=0.0001)

        wss_k = 0  # Sum of distances of samples to their closest cluster center
        for i in range(k):
            cluster_i_x = X[labels==i]
            wss_k += np.sum((np.sum((cluster_i_x - centroids[i])**2, axis=1)))
        inertia.append(wss_k)
        
    return inertia
    
    
# using RFs to compute similarity matrices
def df_scramble(X):
    """Breiman's RF gets X' from X"""
    # X: pd.DataFrame
    X_rand = X.copy()
    for colname in X:
        # duplicate and bootstrap columns of X to get X'
        X_rand[colname] = np.random.choice(X[colname], len(X), replace=True)
    return X_rand


def conjure_twoclass(X : pd.DataFrame)-> (pd.DataFrame, pd.Series):
    """Breiman’s RF conjures up supervised from unsupervised"""
    # Breiman's RF gets X' from X"
    X_rand = df_scramble(X)
    
    # stack X and X'
    X_synth = pd.concat([X, X_rand], axis=0) 
    
    # create y to distinguish X vs X'
    y_synth = np.concatenate([np.zeros(len(X)), np.ones(len(X_rand))], axis=0)
    
    return X_synth, pd.Series(y_synth)


def leaf_samples(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:,t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples


def similarity_matrix(X):
    X_df = pd.DataFrame(X)
    
    # stack [X, X']
    X_synth, y_synth = conjure_twoclass(X_df)
    
    # train RF on X_synth -> y_synth
    rf = RandomForestClassifier()
    rf.fit(X_synth, y_synth)

    # get the similarity matrix
    n = len(X)
    sim_matrix = np.array([[0 for i in range(n)] for j in range(n)])
    leaves_w_records = leaf_samples(rf, X)

    # count how ofter two records appear in same leaf in all trees of forest
    for leaf in leaves_w_records:
        for x_i in leaf:
            for x_j in leaf:
                sim_matrix[x_i][x_j] += 1

    # normalize by number of leaves            
    sim_matrix = sim_matrix / len(leaves_w_records) 
    
    return sim_matrix