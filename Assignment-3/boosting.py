import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:

        summation = np.zeros(len(features))
        for h_t, beta_t in zip(self.clfs_picked, self.betas):
            h_t_preds = h_t.predict(features)
            summation += np.multiply(beta_t, h_t_preds)

        predictions = np.where(summation < 0, -1, 1)
        return predictions.tolist()


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        X = np.array(features)
        y = np.array(labels)
        N = X.shape[0]
        D = X.shape[1]

        # Line 1
        w = np.full(N, 1.0 / N)

        # Line 2
        for t in range(self.T):

            # Line 3
            h_t = None
            min_summation = float("inf")
            for h in self.clfs:
                h_preds = np.array(h.predict(features))
                mismatch = y != h_preds

                summation = np.dot(w, mismatch)

                if summation < min_summation:
                    min_summation = summation
                    h_t = h

            self.clfs_picked.append(h_t)

            # Line 4
            h_t_preds = np.array(h_t.predict(features))
            mismatch = y != h_t_preds
            error = np.dot(w, mismatch)

            # Line 5
            beta_t = (1 / 2.0) * np.log((1 - error) / error)
            self.betas.append(beta_t)

            # Line 6
            for n in range(N):
                if y[n] == h_t_preds[n]:
                    w[n] = w[n] * np.exp(-beta_t)
                else:
                    w[n] = w[n] * np.exp(beta_t)

            # Line 7
            w = w / np.sum(w)

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)


class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        X = np.array(features)
        y = np.array(labels)
        N = X.shape[0]
        D = X.shape[1]

        # Line 1
        p = np.full(N, 1.0 / 2.0)
        self.betas = np.full(N, 1.0 / 2.0)
        f = np.zeros(N)

        # Line 2
        for t in range(self.T):

            # Line 3
            z = (((y + 1) / 2.0) - p) / (p * (1.0 - p))

            # Line 4
            w = np.multiply(p, (1 - p))

            # Line 5
            h_t = None
            min_summation = float("inf")
            for h in self.clfs:
                h_preds = np.array(h.predict(features))
                summation = np.dot(w, np.power((z - h_preds), 2))

                if summation < min_summation:
                    min_summation = summation
                    h_t = h

            self.clfs_picked.append(h_t)

            # Line 6
            h_t_preds = np.array(h_t.predict(features))
            f = f + (1 / 2.0) * h_t_preds

            # Line 7
            p = 1.0 / (1 + np.exp(-2.0 * f))

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
