import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

class PCA:
    def __init__(self, variance_threshold=0.95, n_components=None):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        # Normalizing and centering my data
        X_normalized = np.array(X, dtype=float) / 255.0
        self.mean = np.mean(X_normalized, axis=0)
        X_centered = X_normalized - self.mean
        
        # Computing the sample covariance matrix and eigen value decomposition
        evd_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(evd_matrix)
        
        # Sorting by descending eigenvalues
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_idx]
        eigenvectors_sorted = eigenvectors[:, sorted_idx]
        
        # Determining components for target variance
        self.explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
        cumulative_variance = np.cumsum(self.explained_variance_ratio)

        if self.n_components is None:
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Storing components
        self.components = eigenvectors_sorted[:, :self.n_components]

        return self
        
    def transform(self, X):
        # Normalizing and centering my data
        X_normalized = np.array(X, dtype=float) / 255.0
        X_centered = X_normalized - self.mean
        
        # Projecting to PCA space
        return X_centered @ self.components
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class KNN:
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes = np.unique(y)
        return self

    def _compute_all_distances(self, X):
        X = np.asarray(X)

        # (x - y)^2 = x^2 + y^2 - 2xy
        X2 = np.sum(X**2, axis=1).reshape(-1, 1)
        T2 = np.sum(self.X_train**2, axis=1).reshape(1, -1)
        dist = np.sqrt(np.maximum(X2 + T2 - 2 * X @ self.X_train.T, 0))

        return dist

    def predict(self, X):
        X = np.asarray(X)

        # Vectorized distance computation
        dist = self._compute_all_distances(X)

        # Getting first k nearest neighbors
        idx = np.argpartition(dist, self.k, axis=1)[:, :self.k]
        neighbors = self.y_train[idx]

        # Majority
        return np.array([Counter(row).most_common(1)[0][0] for row in neighbors])

    def predict_proba(self, X):
        X = np.asarray(X)
        dist = self._compute_all_distances(X)

        idx = np.argpartition(dist, self.k, axis=1)[:, :self.k]
        neighbors = self.y_train[idx]

        probs = np.zeros((len(X), len(self.classes)))

        for i, row in enumerate(neighbors):
            for j, cls in enumerate(self.classes):
                probs[i, j] = np.sum(row == cls) / self.k

        return probs
    
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class XGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 lambda_l2=1, gamma=0, min_child_weight=1, threshold=0.5):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_l2 = lambda_l2
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.threshold = threshold
        
        self.trees = []
        self.initial_prediction = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -12, 12)))

    def log_loss_derivatives(self, y_true, y_pred):
        p = self.sigmoid(y_pred)
        grad = p - y_true
        hess = p * (1 - p)
        return grad, hess

    def calc_gain(self, G, H):
        return (G * G) / (H + self.lambda_l2)

    def compute_gain(self, G, H, G_left, H_left, G_right, H_right):
        gain = 0.5 * (self.calc_gain(G_left, H_left) +
                      self.calc_gain(G_right, H_right) -
                      self.calc_gain(G, H)) - self.gamma
        return gain

    def best_split_feature(self, X, grad, hess, feature, sorted_idx):
        fv = X[sorted_idx, feature]

        G_cum = np.cumsum(grad[sorted_idx])
        H_cum = np.cumsum(hess[sorted_idx])

        G_total = G_cum[-1]
        H_total = H_cum[-1]

        best_gain = 0
        best_thr = None

        for i in range(len(sorted_idx) - 1):
            if fv[i] == fv[i + 1]:
                continue

            G_left = G_cum[i]
            H_left = H_cum[i]

            G_right = G_total - G_left
            H_right = H_total - H_left

            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue

            gain = self.compute_gain(G_total, H_total, G_left, H_left, G_right, H_right)

            if gain > best_gain:
                best_gain = gain
                best_thr = (fv[i] + fv[i + 1]) / 2

        return best_gain, best_thr

    def best_split(self, X, grad, hess):
        n_samples, n_features = X.shape

        feat_count = int(n_features)
        features = np.random.choice(n_features, feat_count, replace=False)

        best_gain = 0
        best_feat = None
        best_thr = None

        local_sorted = {f: np.argsort(X[:, f]) for f in features}

        for f in features:
            gain, thr = self.best_split_feature(X, grad, hess, f, local_sorted[f])

            if thr is not None and gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thr = thr

        return best_feat, best_thr

    def build_tree(self, X, grad, hess, depth):
        if (depth >= self.max_depth or len(X) < 2 or hess.sum() < self.min_child_weight):
            leaf_val = -grad.sum() / (hess.sum() + self.lambda_l2)
            return TreeNode(value=leaf_val)

        feat, thr = self.best_split(X, grad, hess)

        if feat is None:
            leaf_val = -grad.sum() / (hess.sum() + self.lambda_l2)
            return TreeNode(value=leaf_val)

        left_mask = (X[:, feat] <= thr)
        right_mask = ~left_mask

        left = self.build_tree(X[left_mask], grad[left_mask], hess[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], grad[right_mask], hess[right_mask], depth + 1)

        return TreeNode(feature_index=feat, threshold=thr, left=left, right=right)

    def predict_tree(self, X, node):
        out = np.zeros(len(X))

        for i, x in enumerate(X):
            cur = node
            while cur.value is None:
                if x[cur.feature_index] <= cur.threshold:
                    cur = cur.left
                else:
                    cur = cur.right
            out[i] = cur.value

        return out

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        p = np.mean(y)
        self.initial_prediction = np.log(p / (1 - p + 1e-12))

        pred = np.full(len(y), self.initial_prediction)

        self.trees = []

        for _ in range(self.n_estimators):

            grad, hess = self.log_loss_derivatives(y, pred)

            tree = self.build_tree(X, grad, hess, depth=0)
            self.trees.append(tree)

            pred += self.learning_rate * self.predict_tree(X, tree)

    def predict_proba(self, X):
        X = np.asarray(X)

        pred = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            pred += self.learning_rate * self.predict_tree(X, tree)

        p = self.sigmoid(pred)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

class SVM:
    def __init__(self, learning_rate=0.001, lambda_p =0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_p = lambda_p
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1/+1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # only regularization penalty
                    self.w -= self.lr * (self.lambda_p * self.w)
                else:
                    # only regularization penalty
                    self.w -= self.lr * (self.lambda_p * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
class RFDecisionTree:
    def __init__(self, max_depth=12, min_samples_split=20, min_samples_leaf=5, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_indices = feature_indices
        self.root = None
    
    def is_leaf_node(self, node):
        return not isinstance(node, dict)

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def gini_gain(self, y, feature_vector, threshold):
        left = feature_vector <= threshold
        right = feature_vector > threshold
        
        if np.sum(left) == 0 or np.sum(right) == 0:
            return -1
            
        n_left, n_right = np.sum(left), np.sum(right)
        n_total = len(y)
        
        gini_left = self.gini_impurity(y[left])
        gini_right = self.gini_impurity(y[right])
        current_gini = self.gini_impurity(y)
        
        gain = current_gini - (n_left/n_total * gini_left + n_right/n_total * gini_right)
        return gain

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        if self.feature_indices is not None:
            features = self.feature_indices
        else:
            features = range(X.shape[1])
            
        for feature in features:
            feature_values = X[:, feature]
            
            if len(feature_values) > 100:
                thresholds = np.percentile(feature_values, [25, 50, 75])
            else:
                unique_values = np.unique(feature_values)
                if len(unique_values) > 10:
                    thresholds = np.percentile(unique_values, [33, 66])
                else:
                    thresholds = unique_values[:-1]
            
            for threshold in thresholds:
                gain = self.gini_gain(y, feature_values, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1 or
            self.gini_impurity(y) < 0.01):
            return self.most_common_label(y)
        
        feature, threshold, gain = self.best_split(X, y)
        
        if gain <= 0 or feature is None:
            return self.most_common_label(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if (np.sum(left_mask) < self.min_samples_leaf or 
            np.sum(right_mask) < self.min_samples_leaf):
            return self.most_common_label(y)
        
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature_index': feature, 
            'threshold': threshold, 
            'left': left_subtree, 
            'right': right_subtree
        }

    def most_common_label(self, y):
        if len(y) == 0:
            return 0
        return Counter(y).most_common(1)[0][0]

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        return self

    def predict_single(self, x, node):
        if self.is_leaf_node(node):
            return node
            
        if x[node['feature_index']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])
    
class RandomForestClassifier:
    def __init__(self, n_estimators=50, max_depth=12, min_samples_split=20, min_samples_leaf=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.n_classes = None

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        self.n_feature_samples = int(np.sqrt(n_features))
        
        self.trees = []
        
        for i in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)

            feature_indices = np.random.choice(n_features, self.n_feature_samples, replace=False)

            tree = RFDecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, 
                min_samples_leaf=self.min_samples_leaf,
                feature_indices=feature_indices
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        return self

    def predict(self, X):        
        X = np.asarray(X)

        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        final_predictions = []
        for sample_idx in range(X.shape[0]):
            sample_predictions = tree_predictions[:, sample_idx]
            majority_vote = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)
    
class KMeans:
    def __init__(self, n_clusters=10, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.cluster_to_label = None

    def _compute_distances(self, X, centroids):
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

    def fit(self, X, y=None):
        X = np.array(X, dtype=float)
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        random_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                members = X[labels == i]
                new_centroids[i] = members.mean(axis=0) if len(members) > 0 else self.centroids[i]
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = np.argmin(self._compute_distances(X, self.centroids), axis=1)

        if y is not None:
            self.cluster_to_label = {}
            for c in range(self.n_clusters):
                members = y[self.labels_ == c]
                self.cluster_to_label[c] = Counter(members).most_common(1)[0][0] if len(members) > 0 else -1

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def predict_labels(self, X):
        if self.cluster_to_label is None:
            raise ValueError("Fit with y first to assign labels.")
        clusters = self.predict(X)
        return np.array([self.cluster_to_label[c] for c in clusters])
    
    def cluster_confidences(kmeans, y_true):
        confidences = {}
        for c in range(kmeans.n_clusters):
            members = y_true[kmeans.labels_ == c]
            if len(members) == 0:
                confidences[c] = 0
            else:
                majority_label, count = Counter(members).most_common(1)[0]
                confidences[c] = count / len(members)
        return confidences
    
class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=100, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        
        self.theta = None
        self.classes = None
        self.n_features = None
        self.loss_history = []

    def _add_bias(self, X):
        if X.ndim == 1:
            X = X[:, np.newaxis]
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
        reg = (self.lambda_reg / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cross_entropy + reg

    def _one_hot_encode(self, y, n_classes):
        y_one_hot = np.zeros((len(y), n_classes))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        X_b = self._add_bias(X)
        m, n = X_b.shape
        y_one_hot = self._one_hot_encode(y, n_classes)

        self.theta = np.random.randn(n, n_classes) * 0.01
        n_batches = max(1, m // self.batch_size)

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y_one_hot[indices]

            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                logits = np.dot(X_batch, self.theta)
                y_pred = self._softmax(logits)

                errors = y_pred - y_batch
                gradients = (1.0 / len(X_batch)) * np.dot(X_batch.T, errors)
                gradients[1:] += (self.lambda_reg / m) * self.theta[1:]

                self.theta -= self.learning_rate * gradients
                epoch_loss += self._compute_loss(y_batch, y_pred)

            self.loss_history.append(epoch_loss / n_batches)

        return self

    def predict_proba(self, X):
        X_b = self._add_bias(np.asarray(X))
        logits = np.dot(X_b, self.theta)
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_parameters(self):
        return {
            'theta': self.theta,
            'classes': self.classes,
            'n_features': self.n_features,
            'loss_history': self.loss_history
        }