from os import path

import numpy as np
import numpy.typing as npt
import sklearn.metrics
from tqdm import tqdm

from helpers import predict, rank_t_score
from multiscale_lbp_descriptor import MultiscaleLBP
from scikit_lbp_descriptor import ScikitLBP


def evaluate_descriptor(descriptor, distance_metric, X: npt.NDArray, y: npt.NDArray,
                        tile_width: int = 0, verbose: bool = True) -> float:
    """
    Function evaluates the provided descriptor using specific distance metric.

    :param descriptor: Feature extractor.
    :param distance_metric: Distance metric
    :param X: A set of data.
    :param y: Corresponding data labels.
    :param verbose: Print results or not.
    :return: Rank 1 accuracy.
    """

    # Use descriptor to encode data
    print("Extracting features...")
    X_features = descriptor.describe(X, tile_width=tile_width)
    print(X_features[0].shape)

    # Compute the distance matrix
    print("Computing distance matrix...")
    distance_matrix = sklearn.metrics.pairwise_distances(X_features, metric=distance_metric)

    # Compute predictions
    print("Computing predictions...")
    y_hat = predict(distance_matrix, y, t=1)

    # Compute rank 1 score
    acc = rank_t_score(y_hat, y)

    if verbose:
        print(f"Rank-{len(y_hat[0])}: {acc}")

    return acc

