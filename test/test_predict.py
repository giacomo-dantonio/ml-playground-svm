import os
import unittest
import predict

class TestTrain(unittest.TestCase):
    def test_load(self):
        """Load an image and extract the features."""

        filepath = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "quattro.png"))
        image = predict.load_image(filepath)

        self.assertEqual((28 * 28, ), image.shape)
        self.assertEqual(0.0, image[0])

