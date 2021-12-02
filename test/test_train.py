import os
import unittest
import train

from test.test_fetch import assert_shapes

class TestTrain(unittest.TestCase):
    __filepath__ = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "test_data.h5"))

    def test_load(self):
        """Load the MNIST split dataset from an HDF5 file."""

        X_train, y_train, X_val, y_val, X_test, y_test = train.load(self.__filepath__)
        assert_shapes(self, X_train, y_train, X_val, y_val, X_test, y_test)
