import numpy as np
from DecisionTree import DecisionTree
from utils import load_data, train_test_split, accuracy


def main():
    
    X, y = load_data("data/winequality-red.csv")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique quality labels: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    tree = DecisionTree(min_samples_split=20, max_depth=6, n_features=None)
    tree.fit(X_train, y_train)

    y_pred_train=tree.predict(X_train)
    train_acc=accuracy(y_train, y_pred_train)
    print(f"Train accuracy:  {train_acc:.4f}")

    y_pred_test = tree.predict(X_test)
    test_acc = accuracy(y_test, y_pred_test)
    print(f"Test accuracy:  {test_acc:.4f}")


if __name__ == "__main__":
    main()