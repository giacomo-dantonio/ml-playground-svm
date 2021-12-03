import argparse
import h5py
import math
import numpy as np
import os
import typing
from numpy.lib.function_base import angle

from scipy.ndimage import interpolation
from sklearn import datasets
from sklearn import model_selection

def augment_dataset(
    dset : np.array,
    labels : np.array,
    shift = True,
    rotate = 0,
    max_rotation = math.pi / 4.0
) -> typing.Tuple[np.array, np.array]:
    augmented_rows = []
    augmented_labels = []
    for (row, label) in zip(dset, labels):
        augmented_rows.append(row)
        augmented_labels.append(label)

        if shift:
            augmented_rows.append(_shift_digit(row, 2, 0))
            augmented_rows.append(_shift_digit(row, -2, 0))
            augmented_rows.append(_shift_digit(row, 0, 2))
            augmented_rows.append(_shift_digit(row, 0, -2))
            augmented_labels.extend(4 * [label])

        if rotate > 0:
            step = max_rotation / rotate
            angles = (n * step for n in range(1, rotate + 1))
            for angle in angles:
                augmented_rows.append(_rotate_digit(row, angle))
                augmented_rows.append(_rotate_digit(row, -angle))
                augmented_labels.extend(2 * [label])

    return np.array(augmented_rows), np.array(augmented_labels)

def download_nist(filepath : str, augment = False):
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

    if augment:
        X_train, y_train = augment_dataset(X_train, y_train, shift=True, rotate=2)
        X_val, y_val = augment_dataset(X_val, y_val, shift=True, rotate=2)
        X_test, y_test = augment_dataset(X_test, y_test, shift=True, rotate=2)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)

        f.create_dataset("X_val", data=X_val)
        f.create_dataset("y_val", data=y_val)

        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)


def _shift_digit(digit : np.array, i : float, j : float) -> np.array:
    image = digit.reshape(28, 28)
    shifted = interpolation.shift(image, [i, j], cval=0)
    return shifted.reshape(digit.shape)


def _rotate_digit(digit : np.array, angle : float) -> np.array:
    image = digit.reshape(28, 28)
    rotated = interpolation.rotate(image, angle, reshape=False)
    return rotated.reshape(digit.shape)


def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch the mnist_784 dataset from openml, and save it to HDF5.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="data.h5",
        help="The filepath where the HDF5 dataset will be saved.")

    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="Augment the dataset by rotating and shifting the images."
    )

    return parser

if __name__ == "__main__":
    parser = _make_argparser()
    args = parser.parse_args()

    if (args.outpath is not None):
        filepath = os.path.abspath(args.outpath)
        download_nist(filepath, args.augment)
