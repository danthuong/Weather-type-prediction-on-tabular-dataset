import numpy as np
from collections import defaultdict
from src.xgboost_tree import XGBoostTree
from src.utils import softmax

class XGBoostManual:
    def __init__(self, n_classes, n_estimators=50, learning_rate=0.1,
                 max_depth=3, min_samples_split=10, reg_lambda=1.0, gamma=0.0,
                subsample=1.0, colsample_bytree=1.0):
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.eta = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        # store list of list: trees[t][k] = tree for round t, class k
        self.trees = []  # length n_estimators, each is list of n_classes trees
        self.init_score = None
        self.feature_importance_ = defaultdict(float)
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def _init_score(self, y):
        # y: (n_samples,) integer labels 0..K-1
        # init logits F0: log(p_k) maybe using class prior
        n = y.shape[0]
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / n
        # avoid zeros
        probs = np.clip(probs, 1e-6, 1-1e-6)
        # initial logit for each class is log(prob)
        # We'll use logit space (un-normalized), consistent with softmax
        F0 = np.log(probs)
        return F0  # shape (n_classes,)

    def fit(self, X, y):
        """
        X: numpy array (n_samples, n_features) dense
        y: numpy array (n_samples,) integer class labels
        """
        n_samples, n_features = X.shape
        K = self.n_classes
        # initialize logits: each sample has K logits
        F0 = self._init_score(y)  # (K,)
        # expand to per-sample
        F = np.tile(F0, (n_samples, 1))  # shape (n, K)
        self.init_score = F0.copy()

        for t in range(self.n_estimators):
            # compute probabilities
            P = softmax(F)  # (n, K)
            # compute gradients and hessians per class
            # for class k: g_i^k = p_i^k - I(y_i==k); h_i^k = p_i^k * (1 - p_i^k)
            trees_this_round = []
            for k in range(K):
                # gradients and hessians for class k
                yk = (y == k).astype(float)
                G = P[:, k] - yk
                H = P[:, k] * (1.0 - P[:, k])
                # fit a regression-like tree to (G,H)
                tree = XGBoostTree(max_depth=self.max_depth,
                                   min_samples_split=self.min_samples_split,
                                   reg_lambda=self.reg_lambda,
                                   gamma=self.gamma, subsample=self.subsample,
                                   colsample_bytree=self.colsample_bytree)
                tree.fit(X, G, H)
                # predict f_k(x) for all samples
                fk = tree.predict(X)  # shape (n,)
                # update global importance
                for feat_idx, gain in tree.feature_importance_.items():
                    self.feature_importance_[feat_idx] += gain
                trees_this_round.append(tree)
                # update logits for that class: F[:,k] += eta * fk
            # after building K trees, update F
            for k in range(K):
                fk = trees_this_round[k].predict(X)
                F[:, k] += self.eta * fk

            self.trees.append(trees_this_round)

        return self

    def predict_logits(self, X):
        """
        compute logits for samples X
        """
        n_samples = X.shape[0]
        k = self.n_classes
        # start with init_score
        F = np.tile(self.init_score, (n_samples, 1))
        for trees_round in self.trees:
            for k, tree in enumerate(trees_round):
                F[:, k] += self.eta * tree.predict(X)
        return F

    def predict_proba(self, X):
        logits = self.predict_logits(X)
        return softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_feature_importance(self, feature_names=None, top_n=20):
        # return sorted importance by gain
        items = sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)
        if feature_names is None:
            return [(str(i), v) for i, v in items[:top_n]]
        else:
            return [(feature_names[i], v) for i, v in items[:top_n]]