import numpy as np

from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax



def decode_labels(encoded_values, column_name, label_encoders):
    le = label_encoders[column_name]
    return le.inverse_transform(encoded_values)

def compute_gradients(F, y_label, n_classes=None):
    """
    F: logits hiện tại của model, shape (n_samples, n_classes)
    y_label: nhãn integer (LabelEncoder), shape (n_samples,)
    n_classes: số lớp (nếu None, tự tính từ y_label)
    
    Trả về:
    G: gradient, shape (n_samples, n_classes)
    H: hessian (xấp xỉ), shape (n_samples, n_classes)
    """

    if n_classes is None:
        n_classes = len(np.unique(y_label))

    # Chuyển y_label thành one-hot encoding
    y_onehot = np.eye(n_classes)[y_label]  # shape (n_samples, n_classes)
    P = softmax(F, axis=1)  # shape (n_samples, n_classes)
    # G_classification = P - y_onehot, nó là residual cho mỗi lớp
    G = P - y_onehot  # shape (n_samples, n_classes)
    # H_classification = P * (1 - P) cho mỗi lớp
    H = P * (1 - P)  # shape (n_samples, n_classes)

    return G, H

def find_best_split(X, G, H, feature_idx, is_categorical, reg_lambda):
    """
    X: (n_samples, n_features)
    G, H: (n_samples,) gradient/hessian cho lớp cụ thể
    feature_idx: index của feature cần split
    categorical: True nếu feature categorical

    Trả về:
    best_gain: gain tốt nhất giữa các class tìm được
    best_val: giá trị threshold (numeric) hoặc category (categorical) tốt nhất giữa các class để split

    Ta duyệt qua tất cả giá trị có của feature để tìm split tốt nhất (exact greedy) giữa các class mục tiêu ta có.
    Vì class mục tiêu là đa lớp, nên mỗi lớp có giá trị gain và threshold/category tốt nhất khác nhau, ta sẽ cần tìm 
    gain tổng của các lớp mục tiêu sao cho là cao nhất, threshold/category của gain cao nhất thì là tốt nhất để split.

    Note: do bộ dataset này chỉ có 13k samples, mỗi sample chỉ có 10 features, không quá lớn nên exact greedy vẫn chấp 
    nhận được dataset mà lớn hơn chút nữa thì phải xài thuật toán khác, không thì chậm lắm
    """
    best_gain = -np.inf
    best_val = None
    if is_categorical:
        # nếu là categorical thì split theo từng giá trị
        for val in np.unique(X[:, feature_idx]):
            left_idx = X[:, feature_idx] == val
            right_idx = ~left_idx
            gain_total = 0
            G_L, H_L = np.sum(G[left_idx], axis=0), np.sum(H[left_idx], axis=0)     # shape = (n_classes)
            G_R, H_R = np.sum(G[right_idx], axis=0), np.sum(H[right_idx], axis=0)
            # gain = left_similarity + right_similarity - parent_similarity
            # similarity = G^2 / (H + reg_lambda)
            gain = (G_L ** 2) / (H_L + reg_lambda) + (G_R ** 2) / (H_R + reg_lambda) - ((G_L + G_R) ** 2) / (H_L + H_R + reg_lambda)
            gain_total = np.sum(gain)
            if gain_total > best_gain:
                best_gain = gain_total
                best_val = val
    else:
        # numeric thì xét tất cả các ngưỡng, tìm ngưỡng tốt nhất
        sorted_idx = np.argsort(X[:, feature_idx])
        X_sorted = X[sorted_idx, feature_idx]
        G_sorted = G[sorted_idx]
        H_sorted = H[sorted_idx]

        G_cumsum = np.cumsum(G_sorted, axis=0)
        H_cumsum = np.cumsum(H_sorted, axis=0)
        G_total = G_cumsum[-1]
        H_total = H_cumsum[-1]

        # chỉ xét threshold giữa các giá trị khác nhau
        unique_vals, indices = np.unique(X_sorted, return_index=True)
        # loại bỏ index đầu tiên vì không split ở cuối
        indices = indices[1:]

        for idx in indices:
            G_L, H_L = G_cumsum[idx - 1], H_cumsum[idx - 1]     # tính tổng index bên trái
            G_R, H_R = G_total - G_L, H_total - H_L             # tính tổng index bên phải
            gain = (G_L ** 2) / (H_L + reg_lambda) + (G_R ** 2) / (H_R + reg_lambda) - ((G_L + G_R) ** 2) / (H_L + H_R + reg_lambda)
            gain_total = np.sum(gain)
            if gain_total > best_gain:
                best_gain = gain_total
                # threshold = trung bình giữa giá trị trước và giá trị tại idx
                best_val = (X_sorted[idx - 1] + X_sorted[idx]) / 2
    return best_gain, best_val

class Node:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.feature_idx = None
        self.threshold = None   # numeric: <=, categorical: ==
        self.left = None
        self.right = None
        self.value = None       # leaf value, vector multi-class
        self.categorical = False

class Tree:
    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma = 0.0):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma

    def fit(self, X, G, H, categorical_cols=[]):
        self.categorical_cols = categorical_cols
        self.n_classes = G.shape[1]
        self.root = self._build_tree(X, G, H, depth=0)

    def _build_tree(self, X, G, H, depth):
        node = Node(depth=depth)
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            node.is_leaf = True
            node.value = np.array([ -np.sum(G[:, k]) / (np.sum(H[:, k]) + self.reg_lambda) for k in range(self.n_classes)])
            return node
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_categorical = False
        
        # duyệt từng feature để tìm giá trị gain và split tốt nhất ứng với feature đó
        # giữa các gain và split đó thì chọn cái tốt nhất làm node cha, rồi tiếp tục
        # chia dữ liệu con để đi đệ quy tiếp
        for f in range(X.shape[1]):
            is_categorical = f in self.categorical_cols
            gain, threshold = find_best_split(X, G, H, f, is_categorical, self.reg_lambda)
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold
                best_categorical = is_categorical
        
        if best_gain < self.gamma:
            node.is_leaf = True
            node.value = np.array([ -np.sum(G[:, k]) / (np.sum(H[:, k]) + self.reg_lambda) for k in range(self.n_classes)])
            return node
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.categorical = best_categorical

        # chia dữ liệu
        if best_categorical:
            left_idx = X[:, best_feature] == best_threshold
        else:
            left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        node.left = self._build_tree(X[left_idx], G[left_idx], H[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], G[right_idx], H[right_idx], depth + 1)

        return node
    
    def predict(self, X):
        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_classes))
        for i in range(n_samples):
            F[i] = self._predict_sample(X[i], self.root)
        return F
    
    def _predict_sample(self, x, node):
        if node.is_leaf:
            return node.value
        if node.categorical:
            if x[node.feature_idx] == node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
        else:
            if x[node.feature_idx] <= node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
            
class XGBoost_Classifier:
    def __init__(self, n_classes, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma=0.0, categorical_cols=[]):
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.categorical_cols = categorical_cols
        self.trees = []
    
    def fit(self, X, y, categorical_cols=[]):
        n_samples = X.shape[0]
        self.n_classes = len(np.unique(y))
        # Khởi tạo F_0 = 0
        F = np.zeros((n_samples, self.n_classes))
        
        for m in range(self.n_estimators):
            print(f"Building tree: {m+1}/{self.n_estimators}")
            G, H = compute_gradients(F, y, self.n_classes)
            tree = Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X, G, H, categorical_cols)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
    
    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_classes))
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        P = softmax(F, axis=1)
        return P