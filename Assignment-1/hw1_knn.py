from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function
        self.features = None
        self.labels = None

    def train(self, features: List[List[float]], labels: List[int]):
        self.features = numpy.asarray(features)
        self.labels = numpy.asarray(labels)

    def predict(self, features: List[List[float]]) -> List[int]:
        X = numpy.asarray(features)

        predictions = []
                
        # For each row
        for point1 in X:

            # Find all distances from the features
            distances = [self.distance_function(point1, point2) for point2 in self.features]
                        
            # Find the k indices with min distances
            idx = numpy.argpartition(distances, self.k)[:self.k]

            # Count the frequency of each label
            counts = numpy.bincount(self.labels[idx])
                        
            # Pick the label with max count
            predictions.append(numpy.argmax(counts))
        
        return predictions

            
if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
