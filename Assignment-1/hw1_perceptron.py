from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        
        w = np.asarray(self.w)
        features = np.asarray(features)
        labels = np.asarray(labels)
        
        converges = True
        
        # loop at most max_iteration times
        for i in range(self.max_iteration):
            
            # shuffle
            p = np.random.permutation(features.shape[0])
            features = features[p]
            labels = labels[p]
            
            converges = True
            
            # loop all features
            for x, y in zip(features, labels):
                
                # misclassified?
                if y * (w.T.dot(x) / (np.linalg.norm(x) + np.finfo(float).eps)) <= self.margin:
                    
                    # update weights
                    w = w + (y * x) / np.linalg.norm(x)
                    converges = False
                
            if converges == True:
                break
        
        self.w = w.tolist()
        
        return converges
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        X = np.asarray(features)
        w = np.asarray(self.w)
        return [1 if result > 0 else -1 for result in X.dot(self.w)]

    def get_weights(self) -> List[float]:
        return self.w
    