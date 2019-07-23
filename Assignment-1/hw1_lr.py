from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.w = None

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        X = numpy.asarray(features)
        X = numpy.insert(X, 0, [1], axis=1)
        y = numpy.asarray(values)
        
        # Check if not invertible
        self.w = numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        X = numpy.asarray(features)
        X = numpy.insert(X, 0, [1], axis=1)
        predictions = X.dot(self.w)
        return predictions

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.w


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.w = None

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        X = numpy.asarray(features)
        X = numpy.insert(X, 0, [1], axis=1)
        y = numpy.asarray(values)

        # Check if not invertible
        self.w = numpy.linalg.inv(X.T.dot(X) + self.alpha * numpy.identity(X.shape[1])).dot(X.T).dot(y)

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        X = numpy.asarray(features)
        X = numpy.insert(X, 0, [1], axis=1)
        predictions = X.dot(self.w)
        return predictions

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.w


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
