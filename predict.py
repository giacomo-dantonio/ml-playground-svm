"""
Assign a class to an image using an SVM model exported with the train module.
"""

import argparse
import joblib
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from PIL import Image
from PIL import ImageOps

_logger = logging.getLogger(__name__)


def load_image(filepath: str) -> np.array:
    """
    Load an image from a file and process it to be used
    with the trained model.
    """
    img = Image.open(filepath)
    img.thumbnail((28, 28))

    gs_img = ImageOps.grayscale(img)
    imgdata = np.array([255.0 - val for val in gs_img.getdata()])

    plt.imsave("last.png", imgdata.reshape((28, 28)), cmap=cm.binary)
    return imgdata


def _make_argparser() -> argparse.ArgumentParser:
    """Construct the parser for the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Predict an image of a digit between 0 and 9 using an SVM classificator.")

    parser.add_argument(
        "images",
        nargs="*",
        help="The filepath of the image to be predicted.")

    parser.add_argument(
        "-m",
        "--model",
        default="model.pkl",
        help="The filepath of the trained model.")

    parser.add_argument(
        "-v",
        "---verbose",
        action="store_true",
        help="Display information about the trained model on stdout."
    )

    return parser

if __name__ == "__main__":
    parser = _make_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s\t%(message)s',
        level=logging.INFO if args.verbose else logging.WARNING
    )

    if not os.path.exists(args.model):
        _logger.warning("Model path %s not found.", args.model)
        exit(1)

    model = joblib.load(args.model)

    for imagepath in args.images:
        if not os.path.exists(imagepath):
            _logger.warning("Image path %s not found.", imagepath)
            continue

        image = load_image(imagepath)

        prediction = model.predict([image])
        print("Predicted value for %s:" % imagepath, prediction[0])
