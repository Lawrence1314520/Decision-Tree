"""
Independent baseline using sklearn's DecisionTreeClassifier
on winequality-red.csv — for comparison with your custom DecisionTree.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    # ---------- 1. Load data ----------
    df = pd.read_csv("data/winequality-red.csv")
    X = df.drop("quality", axis=1).values          # all columns except quality
    y = df["quality"].values                       # target

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique quality labels: {np.unique(y)}")

    # ---------- 2. Train / test split (same style as your code) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.5,
        random_state=42,          # same seed → comparable split
        stratify=y                # keeps class distribution balanced
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ---------- 3. Train sklearn Decision Tree ----------
    # Hyper-parameters chosen to be reasonably close to yours:
    #   min_samples_split=5, max_depth=10
    clf = DecisionTreeClassifier(
        criterion="entropy",      # same impurity measure you used
        min_samples_split=5,
        max_depth=10,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # ---------- 4. Predict & evaluate ----------
    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)

    print(f"\nsklearn DecisionTreeClassifier")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")


if __name__ == "__main__":
    main()