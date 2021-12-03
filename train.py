import argparse
import h5py
import joblib
import json
import logging
import numpy as np
import utils

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm

_logger = logging.getLogger(__name__)

_search_parameters = {
    "rbf": {
        "clf__kernel": ["rbf"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto"],
    },
    "sigmoid": {
        "clf__kernel": ["sigmoid"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto"],
    },
    "poly": {
        "clf__kernel": ["poly"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto"],
        "clf__degree": [2, 3, 5]
    },
}


def load(filepath):
    """
    Loads the MNIST dataset from an HDF5 file, as exported from the `fetch` module.
    """

    _logger.info("Loading dataset from %s", filepath)
    with h5py.File(filepath, "r") as h5_file:
        return (
                np.array(h5_file["X_train"]),
                np.array(h5_file["y_train"]),
                np.array(h5_file["X_val"]),
                np.array(h5_file["y_val"]),
                np.array(h5_file["X_test"]),
                np.array(h5_file["y_test"]),
        )


def grid_search(X, y, kernel="rbf"):
    """
    Perform a grid search on a small validation set and returns
    the best model.
    """

    _logger.info("Performing grid search.")
    with utils.Timer() as t:
        model = _make_pipeline(kernel)

        f1_scorer = metrics.make_scorer(metrics.f1_score, greater_is_better=True, average='micro')
        search = model_selection.GridSearchCV(
            model,
            _search_parameters[kernel],
            verbose=1,
            scoring=f1_scorer,
        )
        search.fit(X, y)

    _logger.info("Search time: {:.2f}s".format(t.elapsed))
    _logger.info("Search Accuracy: %s", search.best_score_)
    _logger.info("Best parameters: %s\n", json.dumps(search.best_params_, indent=2))

    return search.best_estimator_


def train(model, X, y):
    with utils.Timer() as t:
        model.fit(X, y)

    _logger.info("Training time: {:.2f}s".format(t.elapsed))


def _make_pipeline(kernel="rbf", **kwargs):
    return pipeline.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("clf", svm.SVC(kernel=kernel, **kwargs))
    ])


def _show_metrics(model, X_test, y_test):
    """
    Computes the classification metrics for the trained model,
    by applying it to the test set.
    """
    y_predict = model.predict(X_test)
    confusion = metrics.confusion_matrix(y_test, y_predict)
    report = metrics.classification_report(y_test, y_predict)

    return (
        "Metrics computed on the test set\n"
        "{report}\n"
        "Confusion matrix:\n{confusion}\n"
    ).format(
        report = report,
        confusion = confusion
    )


def _make_argparser() -> argparse.ArgumentParser:
    """Construct the parser for the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model from an HDF5 dataset and export it to a file.")

    parser.add_argument(
        "-g",
        "--gridsearch",
        action="store_true",
        help="Perform a grid search for optimizing the hyperparameters. "
            "If specified the model hyperparameters "
            "(e.g. --gamma and --regularization) are ignored.")

    parser.add_argument(
        "-d",
        "--dataset",
        default="data.h5",
        help="A filepath containing the dataset in the HDF5 format.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="model.pkl",
        help="The filepath for storing the trained model.")

    parser.add_argument(
        "-v",
        "---verbose",
        action="store_true",
        help="Display information about the trained model on stdout."
    )

    parser.add_argument(
        "-m",
        "--metrics",
        action="store_true",
        help="Display metrics computed on the test set."
    )

    parser.add_argument(
        "-C",
        "--regularization",
        type=int,
        default=10,
        help="Regularization parameter. "
            "The strength of the regularization is inversely proportional to C. "
            "Must be strictly positive. The penalty is a squared l2 penalty"
    )

    parser.add_argument(
        "-G",
        "--gamma",
        default="auto",
        help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. "
            "Can be a float value or one of 'scale' and 'auto'"
    )

    parser.add_argument(
        "-k",
        "--kernel",
        default="rbf",
        help="The SVM kernel to be used. "
             "The following values are allowed: %s." % ", ".join(_search_parameters.keys())
    )

    return parser

if __name__ == "__main__":
    parser = _make_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s\t%(message)s',
        level=logging.INFO if args.verbose else logging.WARNING
    )

    X_train, y_train, X_val, y_val, X_test, y_test = load(args.dataset)

    if args.gridsearch:
        model = grid_search(X_val, y_val, kernel=args.kernel)
    else:
        try:
            gamma = float(args.gamma)
        except (TypeError, ValueError):
            gamma = args.gamma

        model = _make_pipeline(
            kernel=args.kernel,
            C=args.regularization,
            gamma=args.gamma)

    _logger.info("Training an SVM classifier with %s kernel", args.kernel)
    train(model, X_train, y_train)

    joblib.dump(model, args.outpath)

    if args.metrics:
        _logger.info(_show_metrics(model, X_test, y_test))
