import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
    ClassificationMetrics
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
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()

    X_test = np.asarray(data.test_features, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_samps, n_feats = X_train.shape

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train_s = (X_train - mean) / std

    w = np.zeros(n_feats)
    b = 0
    loss_history: list[float] = []
    eps = 1e-12  #чтобы не брать log(0)

    for i in range(data.epochs):
        p = sigmoid(X_train_s @ w + b)

        p_clipped = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y_train * np.log(p_clipped ) + (1 - y_train) * np.log(1 - p_clipped))
        loss_history.append(float(loss))

        error = p - y_train
        grad_w = X_train_s.T @ error / n_samps
        grad_b = error.sum() / n_samps
        w -= data.learning_rate * grad_w
        b -= data.learning_rate * grad_b

    weights = w / std
    bias = float(b - np.sum(w * mean / std))

    probabilities = sigmoid(X_test @ weights + bias)
    predictions = (probabilities>=0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )

def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1/(1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "decision_tree": DecisionTreeClassifier(random_state=1),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        metrics[name] = ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, proba)),
            accuracy=float(accuracy_score(y_test, pred)),
            precision=float(precision_score(y_test, pred)),
            recall=float(recall_score(y_test, pred)),
            f1=float(f1_score(y_test, pred)),
        )

    return BaselineComparisonResult(
        logistic_regression=metrics["logistic_regression"],
        decision_tree=metrics["decision_tree"],
        knn=metrics["knn"],
    )
