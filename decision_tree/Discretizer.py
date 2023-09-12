import numpy as np


class Clusterizer(object):

    def __init__(self, num_clusters=10):
        """
        Clusterizer class divides data into clusters using k-means clustering algorithm
        :param num_clusters: number of clusters
        """
        assert isinstance(num_clusters, int), "Number of clusters must be an integer"
        assert num_clusters > 0, "Number of clusters should be positive"
        self._num_clusters = num_clusters
        self._means = None

    def __init_clusters(self, data):
        uniques = np.sort(np.unique(data))
        if uniques.size <= self._num_clusters:
            means = uniques
        else:
            dif = uniques.size // self._num_clusters
            indices_for_means = np.array([1 + i * dif for i in range(self._num_clusters)])
            indices_for_means += (uniques.size - indices_for_means[-1]) // 2
            means = np.array([uniques[ind - 1] for ind in indices_for_means])
        clusters = [[] for _ in range(means.size)]
        for elem in uniques:
            pos = Clusterizer.__closest_pos(means, elem)
            clusters[pos].append(elem)
        return clusters

    @staticmethod
    def __closest_pos(arr, x):
        pos = np.searchsorted(arr, x)
        if pos == 0:
            return pos
        elif pos == arr.size:
            return pos - 1
        elif np.abs(arr[pos] - x) < np.abs(arr[pos-1] - x):
            return pos
        return pos - 1

    @staticmethod
    def __get_means_from_clusters(clusters):
        means = []
        for cluster in clusters:
            means.append(sum(cluster) / len(cluster))
        return np.array(means)

    def __update_clusters(self, data, old_clusters):
        """
        Computes new means based on old clusters and then new clusters based on new means
        :param data: data
        :param old_clusters: old clusters
        :return: new clusters corresponding to new means, and a boolean that is True if
        clusters changed
        """
        new_means = Clusterizer.__get_means_from_clusters(old_clusters)
        new_clusters = [[] for _ in range(new_means.size)]
        clusters_changed = False
        for elem in data:
            pos = self.__closest_pos(new_means, elem)
            new_clusters[pos].append(elem)
            if len(old_clusters[pos]) < len(new_clusters[pos]) or \
                    old_clusters[pos][len(new_clusters[pos]) - 1] != new_clusters[pos][-1]:
                clusters_changed = True

        return new_clusters, clusters_changed

    def fit(self, data):
        """
        Fits clusterizer to data, i.e. computes the rule to divide data into clusters
        :param data: data to break into clusters, a 1-dimensional numpy array
        """
        assert isinstance(data, np.ndarray), "data should be a numpy array"
        clusters = self.__init_clusters(data)

        data = np.sort(np.unique(data))
        while True:
            clusters, changed = self.__update_clusters(data, clusters)
            if not changed:
                break
        self._means = Clusterizer.__get_means_from_clusters(clusters)

    def transform(self, X):
        """
        Transforms X according to fitted data into categories: 0, 1, ..., k-1,
        where k=#clusters
        :param X: a 1-dimensional numpy array, data to transform
        :return: transformed data, the same shape as X
        """
        x_transformed = []
        for x in X:
            x_transformed.append(self.__closest_pos(self._means, x))
        return np.array(x_transformed)

class Discretizer(object):

    def __init__(self, num_categories=10):
        assert isinstance(num_categories, int), "Number of categories must be an integer"
        self._num_categories = num_categories
        self._clusterizers = None

    def fit(self, X):
        """
        Fits discretizer to data
        :param X: a numpy array of shape (n_examples, n_features)
        """
        assert isinstance(X, np.ndarray), "X should be a numpy array"
        assert X.ndim == 2, "X should be an array of dimension 2"
        x_t = X.T
        self._clusterizers = []
        for feat in x_t:
            clusterizer = Clusterizer(self._num_categories)
            clusterizer.fit(feat)
            self._clusterizers.append(clusterizer)

    def transform(self, X):
        """
        Transforms X corresponding to fitted data
        :param X: numpy array of dimension 2 with the same features as fitted array
        :return: numpy array of the same shape as X with data broken into categories 0, 1, ..., k,
        where k = #categories-1
        """
        x_t = X.T
        x_transformed = []
        for i, feat in enumerate(x_t):
            x_transformed.append(self._clusterizers[i].transform(feat))
        x_transformed = np.array(x_transformed)
        return x_transformed.T
