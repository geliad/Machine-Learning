import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, _ = k_means.fit(x)

            # Count samples per cluster
            counts = np.bincount(membership)

            # Estimated pi_k
            self.pi_k = counts / np.sum(counts)

            # Estimate variances
            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                # Sample ids from Gaussian k
                idx_for_k = np.where(membership == k)

                # Samples from Gaussian k
                x_ik = x[idx_for_k]

                x_m = x_ik - self.means[k]
                covariance = np.dot(x_m.T, x_m) / counts[k]
                self.variances[k, :, :] = covariance

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means = x[np.random.choice(N, self.n_cluster, False), :]
            self.pi_k = np.full(self.n_cluster, 1.0 / self.n_cluster)
            self.variances = np.array([np.identity(D)] * self.n_cluster)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE

        # Number of means updates
        number_of_updates = 0

        l_old = None
        for iter in range(self.max_iter):

            number_of_updates += 1

            # E-Step - Calculate responsibilities
            responsibilities = self.estep(x)

            # M-Step - Calculate estimated means, variances, pi_k
            self.mstep(x, responsibilities)

            # Compute log likelihood
            l = self.compute_log_likelihood(x)

            if l_old is not None and np.abs(l - l_old) <= self.e:
                break

            l_old = l

        return number_of_updates
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE

        D = self.means.shape[1]
        p_x = np.zeros((N, D))

        Kss = np.random.multinomial(1, self.pi_k, N)
        Ks = np.argmax(Kss, axis=1)

        for i in range(N):
            k = Ks[i]
            p_x[i] = np.random.multivariate_normal(self.means[k], self.variances[k])

        return p_x
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE

        responsibilities = self.compute_responsibilities(x, normalize=False)

        # Compute log likelihood
        sum = np.sum(responsibilities, axis=1)
        log_sum = np.log(sum)
        log_l = np.sum(log_sum)
        return float(log_l)
        # DONOT MODIFY CODE BELOW THIS LINE

    def estep(self, x):
        """
        Calculates posterior probability p(z=k|x), or else responsibilities
        :param x:
        :return:
        """
        responsibilities = self.compute_responsibilities(x, normalize=True)
        return responsibilities

    def mstep(self, x, responsibilities):

        N, D = x.shape

        # Samples in each k
        N_k = np.sum(responsibilities, axis=0)

        # Estimate pi_k p(z=k)
        self.pi_k = N_k / N

        for k in range(self.n_cluster):
            r_k = responsibilities[:, k]

            # Estimate means
            r_kx = r_k * x.T
            r_kx_sum = np.sum(r_kx, axis=1).T
            self.means[k] = (1.0 / N_k[k]) * r_kx_sum

            # Estimate variance
            x_m = x - self.means[k]
            variance = (1.0 / N_k[k]) * np.dot(r_k * x_m.T, x_m)
            self.variances[k, :, :] = variance

    def multivariate_normal_pdf(self, x, mean, sigma):
        """
        Calculate multivariate normal pdf
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Properties
        :param x:
        :param mean:
        :param sigma:
        :return:
        """

        # Fix covariance matrix if not invertible
        while not self.is_invertible(sigma):
            sigma += 0.001 * np.identity(sigma.shape[0])

        det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)

        # Numerator
        x_m = x - mean

        # https://en.wikipedia.org/wiki/Mahalanobis_distance
        squared_mahalanobis_distance = np.einsum('nj,jk,nk->n', x_m, sigma_inv, x_m)

        numerator = np.exp(-0.5 * squared_mahalanobis_distance)

        # Denominator
        two_pi_k = np.power(2 * np.pi, x.shape[1])
        denominator = 1.0 / np.sqrt(two_pi_k * det)

        pdf_x = numerator * denominator
        return pdf_x

    def compute_responsibilities(self, x, normalize=True):
        """
        Compute responsibilities from current values of mean, pi_k and variances
        :param x: the samples
        :param normalize: whether to normalize the final responsibilities values
        :return:
        """
        responsibilities = np.zeros((x.shape[0], self.n_cluster))
        for k in range(self.n_cluster):
            mean = self.means[k]
            sigma = self.variances[k]
            pdf_x = self.multivariate_normal_pdf(x, mean, sigma)
            responsibilities[:, k] = self.pi_k[k] * pdf_x

        if normalize:
            # Normalize each k
            responsibilities_sum = np.sum(responsibilities, axis=1)
            responsibilities = np.divide(responsibilities.T, responsibilities_sum).T

        return responsibilities

    def is_invertible(self, matrix):
        X, Y = matrix.shape
        return X == Y and np.linalg.matrix_rank(matrix) == X
