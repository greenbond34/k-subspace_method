import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def set_mnist(data_dir):
    data, target = fetch_openml("mnist_784", version=1, return_X_y=True)
    data = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=0
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    os.makedirs(data_dir, exist_ok=True)

    np.savez(os.path.join(data_dir, "train.npz"), X_train, y_train)
    np.savez(os.path.join(data_dir, "test.npz"), X_test, y_test)
    print("Created train_test data")


def load_mnist():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data_mnist")

    train_path = os.path.join(data_dir, "train.npz")
    test_path = os.path.join(data_dir, "test.npz")

    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        set_mnist(data_dir)

    train = np.load(train_path, allow_pickle=True)
    X_train = train["arr_0"]
    y_train = train["arr_1"]
    test = np.load(test_path, allow_pickle=True)
    X_test = test["arr_0"]
    y_test = test["arr_1"]

    return X_train, X_test, y_train, y_test


def main() -> None:
    X_train, X_test, y_train, y_test = load_mnist()
    print("Data loaded successfully")
    # ここでX_train, X_test, y_train, y_testを使用するコードを追加できます


if __name__ == "__main__":
    main()
