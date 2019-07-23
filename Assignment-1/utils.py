from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    mse = np.power(y_t - y_p, 2).mean()
    return mse


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    
    predicted_positives = float(np.sum(predicted_labels))
    real_positives = float(np.sum(real_labels))
    
    true_positives = 0
    for label1, label2 in zip(predicted_labels, real_labels):
        if label1 == 1 and label2 == 1:
            true_positives+=1
    
    # Avoid division by zero
    if true_positives == 0 or predicted_positives == 0 or real_positives == 0:
        return 0
    
    # the number of correct positive results divided by the number of all positive results returned by the classifier
    precision =  true_positives / predicted_positives
    
    #  the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive)
    recall =  true_positives / real_positives
    
    # F1 score
    return 2 * (precision * recall) / (precision + recall)


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    """
    input features = [[1, 2, 3]]
    k = 3
    output = [[1, 2, 3, 1, 4, 9, 1, 8, 27]]
    """
    X = np.asarray(features)
    X_prime = np.asarray(X)
    for i in range(2, k+1):
        X_prime=np.concatenate((X_prime, np.power(X, i)), axis=1)    
    return X_prime

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.linalg.norm(np.asarray(point1)-np.asarray(point2))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(np.asarray(point1), np.asarray(point2))


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    return -np.exp(-0.5 *  np.power(euclidean_distance(np.asarray(point1), np.asarray(point2)), 2))

    
class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        def my_func(a):
            inner = np.inner(a,a)         
            if inner != 0:
                return a / np.sqrt(inner)
            return 0

        return np.apply_along_axis(my_func, 1, features)


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min = None
        self.max = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        min [-1, -1]
        max [2, 5]
        """
        
        X = np.asarray(features)
        
        # Find min of each feature
        if self.min is None:
            self.min = X.min(axis=0)
            
        # Find max of each feature
        if self.max is None:
            self.max = X.max(axis=0)
            
        X_prime = (X - self.min) / (self.max - self.min) 
        
        return X_prime
