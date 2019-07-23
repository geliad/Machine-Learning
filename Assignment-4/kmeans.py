import numpy as np


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE

        # Initialize means by picking self.n_cluster from N data points
        means = x[np.random.choice(N, self.n_cluster, replace=False), :]

        membership = None

        # Number of means updates
        number_of_updates = 0

        J = None

        for iter in range(self.max_iter):
            squared_euclidean_distances = np.zeros([N, self.n_cluster])

            # Compute squared euclidean distances for each k
            for k in range(self.n_cluster):
                euclidean_distances = np.linalg.norm(means[k] - x, axis=1)
                squared_euclidean_distances[:, k] = np.power(euclidean_distances, 2)

            # Compute membership r_ik
            membership = np.argmin(squared_euclidean_distances, axis=1)

            # Compute distortion measure J_new
            r_ik_distance = np.choose(membership, squared_euclidean_distances.T)
            J_new = (1.0 / N) * np.sum(r_ik_distance)

            if J is not None and np.abs(J - J_new) <= self.e:
                break

            J = J_new

            # Update means
            for k in range(self.n_cluster):
                idx_for_k = np.where(membership == k)

                # Update if there are samples members of this mean
                if len(idx_for_k) > 0:
                    means[k] = np.average(x[idx_for_k], axis=0)

            number_of_updates += 1

        return means, membership, number_of_updates
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = k_means.fit(x)
        centroid_labels = np.zeros(self.n_cluster)
        for k in range(self.n_cluster):
            idx_for_k = np.where(membership == k)

            if len(idx_for_k) > 0:
                labels = y[idx_for_k]
                counts = np.bincount(labels)
                centroid_labels[k] = np.argmax(counts)
            else:
                centroid_labels[k] = 0

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        euclidean_distances = np.zeros([N, self.n_cluster])

        # Compute euclidean distances for each k
        for k in range(self.n_cluster):
            euclidean_distances[:, k] = np.linalg.norm(self.centroids[k] - x, axis=1)

        # Compute nearest neighbors
        nn = np.argmin(euclidean_distances, axis=1)

        return self.centroid_labels[nn]

        # DONOT CHANGE CODE BELOW THIS LINE
