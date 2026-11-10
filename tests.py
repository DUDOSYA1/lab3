import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import RegressionInput
from linear_models import train_linear_regression

df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

features = [c for c in df.columns if c != "Performance Index"]
X = df[features].to_numpy(float)
y = df["Performance Index"].to_numpy(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

res = train_linear_regression(RegressionInput(X_tr, y_tr, X_te, y_te))
test_mse = float(np.mean((res.predictions - y_te) ** 2))

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
