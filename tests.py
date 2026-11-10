import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import train_linear_regression, train_logistic_classifier

df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

features = [c for c in df.columns if c != "Performance Index"]
X = df[features].to_numpy(float)
y = df["Performance Index"].to_numpy(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

res = train_linear_regression(RegressionInput(X_tr, y_tr, X_te, y_te))
test_mse = float(np.mean((res.predictions - y_te) ** 2))

print("ЛИНЕЙНАЯ РЕГРЕССИЯ")
print()

print("Коэффициенты:")
for name, w in zip(features, res.weights):
    print(f"  {name} {w}")
print(f"  {'bias'} {res.bias}")
print()

h = res.loss_history
print("MSE по эпохам:")
for e in (1, 10, 50, 100, len(h)):
    print(f"  Эпоха {e}: {h[e - 1]}")
print()

print(f"train MSE: {h[-1]}")
print(f"test MSE: {test_mse}")

#-----------------------------------------------------------------------------------------

credit = pd.read_csv("german.csv", sep=";")
target = credit.columns[0]  # Creditability
c_features = list(credit.columns[1:])
Xc = credit[c_features].to_numpy(float)
yc = credit[target].to_numpy(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=1, stratify=yc)

clf = train_logistic_classifier(ClassificationInput(Xc_tr, yc_tr, Xc_te, yc_te))
metrics = {
    "roc_auc": float(roc_auc_score(yc_te, clf.probabilities)),
    "accuracy": float(accuracy_score(yc_te, clf.predictions)),
    "precision": float(precision_score(yc_te, clf.predictions)),
    "recall": float(recall_score(yc_te, clf.predictions)),
    "f1": float(f1_score(yc_te, clf.predictions)),
}

print()
print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print()

print(f"train: {len(Xc_tr)}, test: {len(Xc_te)}, доля класса 1 в test: {yc_te.mean()}")
print()

print("Коэффициенты:")
for name, w in zip(c_features, clf.weights):
    print(f"  {name} {w}")
print(f"  {'bias'} {clf.bias}")
print()

h = clf.loss_history
print("BCE на train по эпохам:")
for e in (1, 10, 100, 500, len(h)):
    print(f"  эпоха {e}: {h[e - 1]}")
print()

print("Метрики на test:")
for name, value in metrics.items():
    print(f"  {name:10s} {value:.4f}")
