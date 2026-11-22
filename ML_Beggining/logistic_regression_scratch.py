"""Multiclass logistic regression implemented with NumPy and gradient descent.

For each class, the model calculates a score ``X @ weights + bias`` and turns
the scores into probabilities with softmax. It minimizes cross-entropy loss:

    loss = -mean(sum(one_hot(y) * log(probabilities)))
           + l2 * sum(weights ** 2) / 2

Example:
    model = LogisticRegressionScratch()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

import numpy as np


class LogisticRegressionScratch:
    """Logistic regression classifier for binary and multiclass problems."""

    def __init__(
        self,
        learning_rate=0.1,
        n_iterations=5000,
        l2=0.01,
        tol=1e-6,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be greater than 0.")
        if l2 < 0:
            raise ValueError("l2 must be non-negative.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.tol = tol

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (samples, features).")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numeric values.")
        return X

    @staticmethod
    def _softmax(scores):
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, X, y):
        """Fit model parameters and return this estimator."""
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of class labels.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("y must contain at least two distinct classes.")

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        X_scaled = (X - self.mean_) / self.scale_

        n_samples, n_features = X_scaled.shape
        n_classes = self.classes_.size
        targets = np.eye(n_classes)[encoded_y]
        self.weights_ = np.zeros((n_features, n_classes), dtype=float)
        self.bias_ = np.zeros(n_classes, dtype=float)

        for iteration in range(self.n_iterations):
            probabilities = self._softmax(X_scaled @ self.weights_ + self.bias_)
            errors = probabilities - targets
            weight_gradient = (X_scaled.T @ errors) / n_samples
            weight_gradient += self.l2 * self.weights_
            bias_gradient = np.mean(errors, axis=0)

            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient

            if max(
                np.max(np.abs(weight_gradient)),
                np.max(np.abs(bias_gradient)),
            ) <= self.tol:
                break

        self.n_iter_ = iteration + 1
        self.loss_ = self._loss(X_scaled, targets)
        return self

    def _loss(self, X_scaled, targets):
        probabilities = self._softmax(X_scaled @ self.weights_ + self.bias_)
        log_probabilities = np.log(np.clip(probabilities, 1e-15, 1.0))
        cross_entropy = -np.mean(np.sum(targets * log_probabilities, axis=1))
        return cross_entropy + 0.5 * self.l2 * np.sum(self.weights_ ** 2)

    def predict_proba(self, X):
        """Return class probabilities in the order of ``classes_``."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before predict_proba.")
        X = self._prepare_X(X)
        if X.shape[1] != self.mean_.size:
            raise ValueError("X must have the same number of features used in fit.")
        X_scaled = (X - self.mean_) / self.scale_
        return self._softmax(X_scaled @ self.weights_ + self.bias_)

    def predict(self, X):
        """Predict the most likely class for each sample."""
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[class_indices]

    def score(self, X, y):
        """Return the mean classification accuracy."""
        y = np.asarray(y)
        predictions = self.predict(X)
        if y.ndim != 1 or y.shape[0] != predictions.shape[0]:
            raise ValueError("y must be 1D and match the number of samples in X.")
        return np.mean(predictions == y)
