from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    
    k=1  # nearest neighbors
    
    [N,d] = train_image_feats.shape
    M = test_image_feats.shape[0]
    train_labels = np.array(train_labels)
    test_predicts = []
    
    dista = distance.cdist(train_image_feats,test_image_feats,'euclidean')
    KNN = np.argsort(dista,axis=0)
    for i in range(M):  
        labels_unique,counts = np.unique(train_labels[KNN[0:k,i]],return_counts = True)
        test_predicts.append(labels_unique[np.argmax(counts)])
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
