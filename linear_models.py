import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features,dtype=float)
    y_train = np.asarray(data.train_target,dtype=float).ravel()

    X_test = np.asarray(data.test_features,dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_samps, n_feats = X_train.shape

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1 #для случая конст признака
    X_train_s = (X_train-mean)/std

    w = np.zeros(n_feats)
    b = 0
    loss_history = []

    for i in range(data.epochs):
        y_pred = X_train_s @ w + b
        error = y_pred - y_train
        loss_history.append(float(np.mean(error**2)))

        grad_w = 2 / n_samps * (X_train_s.T @ error)
        grad_b = 2 / n_samps * error.sum()
        w -= data.learning_rate * grad_w
        b -= data.learning_rate * grad_b

    weights = w / std
    bias = float(b - np.sum(w * mean / std))

    predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    raise NotImplementedError


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    raise NotImplementedError
