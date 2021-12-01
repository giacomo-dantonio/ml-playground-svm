import h5py
import numpy as np

def load(filepath):
    """
    Loads the MNIST dataset from an HDF5 file, as exported from the `fetch` module.
    """

    with h5py.File(filepath, "r") as h5_file:
        return (
                np.array(h5_file["X_train"]),
                np.array(h5_file["y_train"]),
                np.array(h5_file["X_val"]),
                np.array(h5_file["y_val"]),
                np.array(h5_file["X_test"]),
                np.array(h5_file["y_test"]),
        )
