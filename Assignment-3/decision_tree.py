import numpy as np
from typing import List
from classifier import Classifier


class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert (len(features) > 0)

        self.feautre_dim = len(features[0])
        num_cls = np.max(labels) + 1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name='  ' + name + '/' + str(idx_child), indent=indent + '  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent + '}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label  # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the dim of feature to be splitted

        self.feature_uniq_split = None  # the feature to be splitted

    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array, 
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of
                      corresponding training samples
                      e.g. check Piazza
                      branches = [[2,2], [4,0]]
            '''
            denominators = np.sum(branches, axis=0)
            probabilities = np.divide(branches, denominators)

            log_probabilities = probabilities.copy()
            log_probabilities[probabilities == 0] = 1
            log_probabilities = np.log2(log_probabilities)

            entropy = -np.sum(np.multiply(probabilities, log_probabilities), axis=0)

            weighted_avg = denominators / np.sum(denominators)
            cond_entropy = float(np.sum(np.multiply(weighted_avg, entropy)))
            return cond_entropy

        ############################################################
        # compare each split using conditional entropy
        #       find the best split
        ############################################################
        min_cond_entropy = np.inf
        np_features = np.array(self.features)

        # The unique labels
        labels_uniq, l_indices = np.unique(self.labels, return_inverse=True)

        for idx_dim in range(len(self.features[0])):

            # Extract feature values
            feature = np_features[:, idx_dim]

            # The unique feature values
            feature_uniq_split, f_indices = np.unique(feature, return_inverse=True)

            # Construct branches
            branches = np.zeros((len(labels_uniq), len(feature_uniq_split)), dtype=int)
            for i, j in zip(l_indices, f_indices):
                branches[i][j] += 1

            # Compute conditional entropy
            ce = conditional_entropy(list(branches.tolist()))

            # Pick the best split
            if ce < min_cond_entropy:
                min_cond_entropy = ce
                self.dim_split = idx_dim
                self.feature_uniq_split = feature_uniq_split.tolist()

        ############################################################
        # split the node, add child nodes
        ############################################################

        # Delete column
        # np_features = np.delete(np_features, self.dim_split, 1)
        # feature = np_features[:, self.dim_split]

        for i in range(len(self.feature_uniq_split)):
            features = []
            labels = []

            for f, other_features, l in zip(np_features[:, self.dim_split], np.delete(np_features, self.dim_split, 1), self.labels):
                if f == self.feature_uniq_split[i]:
                    features.append(other_features.tolist())
                    labels.append(l)

            child = TreeNode(features, labels, len(np.unique(labels)))
            if len(features[0]) == 0:
                child.splittable = False
            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max
