import numpy as np
import warnings as wrn
from decision_tree import Discretizer as disc


class Node(object):
    def __init__(self, feat: int = None, feat_val=None, left=None, right=None, res: int = None):
        """

        :param feat: feature, on which to divide
        :param feat_val: threshold value
        :param left: left child, with feature values below the threshold
        :param right: right child, with feature values above the threshold
        :param res: predicted class, if the node is terminal; None, otherwise
        """
        self.feat = feat
        self.left = left
        self.right = right
        self.res = res
        self.feat_val=feat_val


class DecisionTreeClassifier(object):
    def __init__(self, max_depth=-1):
        """
        Class performing binary classification using decision tree algorithm
        :param max_depth: maximal depth of the tree
        """
        assert isinstance(max_depth, int), "max_depth should be an integer"
        self._tree = None
        self._discretizer = None
        self._max_depth = max_depth

    @staticmethod
    def __prevalence_data(y):
        """

        :param y: a vector of zeros and ones
        :return: return the prevalent value and respective number of occurrences
        """
        num_ones = 0
        for val in y:
            num_ones += val
        if num_ones > y.shape[0] // 2:
            return 1, num_ones
        else:
            return 0, y.shape[0] - num_ones

    def __feat_acc(self, x_feat, x_feat_val, y):
        """
        Computes how well does the feature divide the data on a given value
        :param x_feat: prospective feature for division
        :param x_feat_val: prospective feature value
        :param y: vector of targets
        :return: choosing prevalent values, how many correct answers would you score?
                 None, if the feature either always or never equals x_feat_val
        """
        prevalent_val_1, prevalent_freq_1 = self.__prevalence_data(y[x_feat == x_feat_val])
        prevalent_val_0, prevalent_freq_0 = self.__prevalence_data(y[x_feat != x_feat_val])
        if prevalent_freq_0 == 0 or prevalent_freq_1 == 0:
            return None
        return prevalent_freq_0 + prevalent_freq_1

    def __best_attr(self, data, feats_left):
        X, y = data
        best_feat = None
        best_feat_val = None
        best_acc = None
        for feat_num, feat in enumerate(feats_left):
            for feat_val in feat:
                acc = self.__feat_acc(X[feat_num], feat_val, y)
                if acc is not None:
                    if best_feat is None or acc > best_acc:
                        best_feat = feat_num
                        best_feat_val = feat_val
                        best_acc = acc
        return best_feat, best_feat_val

    @staticmethod
    def __partition(data, feat, feat_val):
        X, y = data
        data_true = X[:, X[feat] == feat_val], y[X[feat] == feat_val]
        data_false = X[:, X[feat] != feat_val], y[X[feat] != feat_val]
        return data_true, data_false

    def __process(self, data, feats_left, cur_depth):
        X, y = data
        prevalent_val, prevalent_freq = DecisionTreeClassifier.__prevalence_data(y)
        if feats_left is None:
            return Node(res=prevalent_val)
        if prevalent_freq == 0 or prevalent_freq == y.shape[0]:
            return Node(res=prevalent_val)
        if cur_depth == self._max_depth:
            return Node(res=prevalent_val)
        f, val = self.__best_attr(data, feats_left)
        if f is None:
            return Node(res=prevalent_val)
        data_true, data_false = DecisionTreeClassifier.__partition(data, f, val)
        feats_left_true = feats_left.copy()
        feats_left_true.pop(f)
        feats_left_false = feats_left.copy()
        feats_left_false[f] = feats_left_false[f][feats_left_false[f] != val]
        if feats_left_false[f].size < 2:
            feats_left_false.pop(f)
        return Node(feat=f, feat_val=val,
                    left=self.__process(data_true, feats_left_true, cur_depth+1),
                    right=self.__process(data_false, feats_left_false, cur_depth+1))

    @staticmethod
    def __max_val(a):
        max_val= 0
        for val in a:
            if val > max_val:
                max_val = val
        return max_val

    def fit(self, X, y):
        """
        Fits decision tree to data
        :param X: numpy array of shape (n_examples, n_features) - data
        :param y: numpy array of shape (n_examples, 1), contains 0s and 1s - labels
        """
        if not isinstance(X, np.ndarray):
            wrn.warn("X is not a numpy array. Behaviour is undefined")
        if not isinstance(y, np.ndarray):
            wrn.warn("y is not a numpy array. Behaviour is undefined")
        assert X.ndim == 2 and y.ndim == 2, "X and y should be numpy arrays of dimension 2"
        assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"
        assert y.shape[1] == 1, "y should have exactly 1 column"

        self._discretizer = disc.Discretizer(2)
        self._discretizer.fit(X)
        x = self._discretizer.transform(X)
        x = x.T
        all_feats = []
        for feat in x:
            max_val = DecisionTreeClassifier.__max_val(feat)
            all_feats.append(np.array(range(max_val + 1)))
        self._tree = self.__process(data=(x, y), feats_left=all_feats,
                                    cur_depth=0)

    def __iterate_tree(self, tree, x):
        if tree.res is not None:
            return tree.res
        if x[tree.feat] == tree.feat_val:
            return self.__iterate_tree(tree.left, x)
        else:
            return self.__iterate_tree(tree.right, x)

    def predict(self, X):
        """
        Perform binary classification for each example in X
        :param X: numpy array of shape (n_examples, n_feature) - data
        :return: numpy array of shape (n_examples, 1) of 0s and 1s - prediction for each example
        """
        if not isinstance(X, np.ndarray):
            wrn.warn("X is not a numpy array. Behaviour is undefined")
        assert X.ndim == 2, "X should be a numpy array of dimension 2"
        X_transformed = self._discretizer.transform(X)
        ret = np.array([self.__iterate_tree(self._tree, x) for x in X_transformed])
        ret.shape = (-1, 1)
        return ret
