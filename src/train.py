"""Diabetes progression regression analysis.

Trains Linear Regression and Ridge regression on the scikit-learn
diabetes dataset and prints MAE, RMSE and R-squared for both models.
Also saves a residual plot for the Ridge model to outputs/.

For learning and portfolio demonstration only — not clinical or
medical advice.

Usage:
    python src/train.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="progression")
    print("Dataset shape:", X.shape)
    print("Target (progression) summary:")
    print(y.describe().round(2))
    print("\nMissing values:", int(X.isna().sum().sum()))

    corr = pd.concat([X, y], axis=1).corr()["progression"].drop("progression")
    print("\nCorrelation with progression, strongest first:")
    print(corr.sort_values(key=abs, ascending=False).round(3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"--- {name} ---")
        print("MAE :", round(mean_absolute_error(y_test, pred), 2))
        print("RMSE:", round(float(np.sqrt(mean_squared_error(y_test, pred))), 2))
        print("R^2 :", round(r2_score(y_test, pred), 4))

    best = Ridge(alpha=1.0).fit(X_train, y_train)
    pred = best.predict(X_test)
    resid = y_test - pred
    print("\nResidual mean (Ridge):", round(resid.mean(), 2))
    print("Residual std  (Ridge):", round(resid.std(), 2))

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.scatter(pred, resid, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted progression")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residual plot — Ridge regression")
    plt.tight_layout()
    plt.savefig("outputs/residual_plot.png")
    print("Saved: outputs/residual_plot.png")


if __name__ == "__main__":
    main()
