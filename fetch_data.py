import argparse
import h5py
import numpy as np
import os

from sklearn import datasets
from sklearn import model_selection

def download_nist(filepath : str):
    """
        Fetch the mnist_784 dataset from openml (https://www.openml.org/d/554).
        Splits it into training, validation, and test set.
        And write it to `filepath` in the hdf5 format.
    """

    mnist = datasets.fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X = X.to_numpy()
    y = y.astype(np.uint8)

    # The MNIST dataset is already split into a training set (the first 60000)
    # and a test set (the last 10000).
    # We split the training set once again for model validation.

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=10000)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)

        f.create_dataset("X_val", data=X_val)
        f.create_dataset("y_val", data=y_val)

        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)

def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch the mnist_784 dataset from openml, and save it to HDF5.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="data.h5",
        help="The filepath where the HDF5 dataset will be saved.")

    return parser

if __name__ == "__main__":
    parser = _make_argparser()
    args = parser.parse_args()

    if (args.outpath is not None):
        filepath = os.path.abspath(args.outpath)
        download_nist(filepath)
