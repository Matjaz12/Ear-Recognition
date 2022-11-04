import numpy.typing as npt

import numpy as np


def compute_concatenated_histogram(image, tile_size, num_bins):
    """
    Function computes a concatenated histogram, by spliting the image
    into tiles of size `tile_size` and computing a local histogram on each tile.
    Histograms are normalized and concatenated together.

    :param image: Input image
    :param tile_size: Size of each tile (Width, Height)
    :param num_bins: Number of bins used in computation of local histogram.
    :return: Concatenated histogram.
    """

    height, width = image.shape
    tile_height, tile_width = tile_size
    fv = []

    # Split image into tiles of size (tile_height x tile_width)
    tiled_image = image.reshape(height // tile_height,
                                tile_height,
                                width // tile_width,
                                tile_width)

    tiled_image = tiled_image.swapaxes(1, 2)
    tiled_image = tiled_image.reshape(tiled_image.shape[0] * tiled_image.shape[1], tile_height, tile_width)

    # Compute histogram for each tile and concatenate them
    for tile in tiled_image:
        from scipy.sparse import csr_matrix
        tile_hist, _ = np.histogram(tile, density=True, bins=num_bins, range=(0, num_bins))
        fv.extend(tile_hist)

    # Normalize the concatenated histogram
    fv = np.array(fv)
    # fv = fv / np.sum(fv)

    return fv


def predict(distance_matrix: npt.NDArray, y: npt.NDArray, t: int = 1) -> npt.NDArray:
    """
    Function takes in a distance_matrix of size (`num_samples x num_samples`), it finds
    the `t` closest vectors for each vector (and assigns their corresponding labels).

    :param distance_matrix: A matrix holding vector pair wise distances.
    :param y: Vector of labels of size (`num_samples`)
    :param t: Number of closest vectors to consider
    :return: Vector of predictions `y_hat` of size (`num_samples x t`).
    """
    y_hat = np.zeros((len(y), t))

    for i in range(len(distance_matrix)):
        distance_matrix_i = distance_matrix[i].copy()
        distance_matrix_i[i] = np.inf

        closest_indices = np.argsort(distance_matrix_i)[0: t]
        closest_labels = y[closest_indices]
        y_hat[i] = closest_labels

    return y_hat


def rank_t_score(y_hat: npt.NDArray, y: npt.NDArray) -> float:
    """
    Function computes the rank t accuracy.

    :param y_hat: Vector of predictions `y_hat` of size (`num_samples x t`).
    :param y: Vector of labels of size (`num_samples`).
    :return: Rank t accuracy.
    """
    correct = 0

    for i in range(len(y)):
        if y[i] in y_hat[i]:
            correct += 1

    return correct / len(y)
