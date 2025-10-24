from collections import defaultdict
import numpy as np
from src.TreeNode import TreeNode
class XGBoostTree:
    """
    Single regression tree used inside XGBoost (predicts residuals / logits increment).
    Builds splits by maximizing Gain = formula in paper.
    """
    def __init__(self, max_depth=3, min_samples_split=10, reg_lambda=1.0, gamma=0.0, subsample=1.0, colsample_bytree=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None
        # feature importance by gain
        self.feature_importance_ = defaultdict(float)
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def _calc_weight(self, G, H):
        # w* = - G / (H + lambda)
        return - G / (H + self.reg_lambda)

    def _calc_gain(self, GL, HL, GR, HR):
        # Gain formula for splitting into left/right (1/2 factor)
        # Gain = 1/2 * (GL^2/(HL+λ) + GR^2/(HR+λ) - (GL+GR)^2/(HL+HR+λ)) - γ
        eps = 1e-12
        left_term = (GL**2) / (HL + self.reg_lambda + eps)
        right_term = (GR**2) / (HR + self.reg_lambda + eps)
        total_term = ((GL + GR)**2) / (HL + HR + self.reg_lambda + eps)
        gain = 0.5 * (left_term + right_term - total_term) - self.gamma
        return gain

    def _build_node(self, X, G, H, depth):
        node = TreeNode(depth=depth)
        # compute sums for this node
        G_sum = G.sum()
        H_sum = H.sum()
        node.weight = self._calc_weight(G_sum, H_sum)
        n_samples, n_features = X.shape
        
        # 1️⃣ Row subsampling (chọn ngẫu nhiên một phần mẫu để huấn luyện node này)
        if hasattr(self, "subsample") and self.subsample < 1.0:
            idx = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
            X, G, H = X[idx], G[idx], H[idx]
            n_samples = X.shape[0]

        # 2️⃣ Feature subsampling (chọn ngẫu nhiên một phần đặc trưng)
        features = np.arange(n_features)
        if hasattr(self, "colsample_bytree") and self.colsample_bytree < 1.0:
            features = np.random.choice(features, int(self.colsample_bytree * n_features), replace=False)

        # stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            node.is_leaf = True
            return node

        best_gain = -np.inf
        best_feat = None
        best_thr = None
        best_left_idx = None
        best_right_idx = None
        # iterate features and candidate thresholds
        for feat in features:
            xs = X[:, feat]
            # sort by feature
            order = np.argsort(xs)
            xs_sorted = xs[order]
            G_sorted = G[order]
            H_sorted = H[order]

            # candidate splits: between unique values
            # to speed up, consider unique values and midpoints
            unique_vals = np.unique(xs_sorted)
            if unique_vals.shape[0] == 1:
                continue
            # thresholds as midpoints between consecutive unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            # to compute GL/HL efficiently, iterate in sorted order and accumulate
            # We'll accumulate by index positions matching thresholds
            idx = 0
            GL = 0.0
            HL = 0.0
            # We need mapping from value -> positions; but simpler: iterate positions and test at boundaries
            # Build prefix sums for G_sorted and H_sorted
            pref_G = np.cumsum(G_sorted)
            pref_H = np.cumsum(H_sorted)
            # For each threshold candidate, find last position <= threshold
            for thr in thresholds:
                # find rightmost index where xs_sorted <= thr
                # use binary search on xs_sorted
                import bisect
                pos = bisect.bisect_right(xs_sorted, thr)  # count of items <= thr
                if pos == 0 or pos == n_samples:
                    continue
                GL = pref_G[pos-1]
                HL = pref_H[pos-1]
                GR = G_sum - GL
                HR = H_sum - HL
                gain = self._calc_gain(GL, HL, GR, HR)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr
                    # store indices for split
                    left_idx = order[:pos]
                    right_idx = order[pos:]
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        # if best_gain not positive, make leaf
        if best_gain <= 0 or best_feat is None:
            node.is_leaf = True
            return node

        # else create internal node
        node.is_leaf = False
        node.feature_idx = int(best_feat)
        node.threshold = float(best_thr)
        node.gain = float(best_gain)
        # accumulate feature importance
        self.feature_importance_[node.feature_idx] += best_gain

        # recursively build children
        left_X = X[best_left_idx]
        right_X = X[best_right_idx]
        left_G = G[best_left_idx]
        right_G = G[best_right_idx]
        left_H = H[best_left_idx]
        right_H = H[best_right_idx]

        node.left = self._build_node(left_X, left_G, left_H, depth + 1)
        node.right = self._build_node(right_X, right_G, right_H, depth + 1)
        return node

    def fit(self, X, G, H):
        """
        X: (n_samples, n_features)
        G: (n_samples,) gradients
        H: (n_samples,) hessians
        """
        self.root = self._build_node(X, G, H, depth=0)

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.weight
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        # return vector of predictions f(x) for each sample
        preds = np.array([self._predict_one(x, self.root) for x in X])
        return preds