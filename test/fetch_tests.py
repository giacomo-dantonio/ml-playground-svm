import fetch
import h5py
import os
import unittest

def assert_shapes(suite, X_train, y_train, X_val, y_val, X_test, y_test):
    suite.assertEqual((50000, 784), X_train.shape)
    suite.assertEqual((50000,), y_train.shape)

    suite.assertEqual((10000, 784), X_val.shape)
    suite.assertEqual((10000,), y_val.shape)

    suite.assertEqual((10000, 784), X_test.shape)
    suite.assertEqual((10000,), y_test.shape)

class TestFetch(unittest.TestCase):
    __filepath__ = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "data.h5"))

    def setUp(self):
        self.delete_model()

    def tearDown(self):
        self.delete_model()

    def delete_model(self):
        if os.path.exists(self.__filepath__):
            os.remove(self.__filepath__)

    def test_fetch(self):
        """Fetch the MNIST dataset and save it to a local file."""
        fetch.download_nist(self.__filepath__)

        self.assertTrue(os.path.exists(self.__filepath__))
        with h5py.File("data.h5", "r") as h5_file:
            keys = h5_file.keys()
            self.assertIn("X_train", keys)
            self.assertIn("y_train", keys)
            self.assertIn("X_val", keys)
            self.assertIn("y_val", keys)
            self.assertIn("X_test", keys)
            self.assertIn("y_test", keys)

            assert_shapes(
                self,
                h5_file["X_train"],
                h5_file["y_train"],
                h5_file["X_val"],
                h5_file["y_val"],
                h5_file["X_test"],
                h5_file["y_test"],
            )

if __name__ == '__main__':
    unittest.main()