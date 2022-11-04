import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import bitstring
from numba import njit
from sympy import divisors
from tqdm import tqdm

from helpers import compute_concatenated_histogram
from load_data import load_data
from scikit_lbp_descriptor import ScikitLBP


@njit
def get_val(image: npt.NDArray, coord: Tuple[int, int], default_val: float = 0.0) -> float:
    """
    Indexes an array by checking the bounds, if indices
    out of bounds it returns the default value.

    :param image: Input image
    :param coord: indices [i, j]
    :param default_val: Value to be returned if indices are out of bounds.
    :return: Pixel value.
    """
    height, width = image.shape
    i, j = coord

    if i >= height or i < 0 or j >= width or j < 0:
        val = default_val
    else:
        val = image[i][j]

    return val


@njit
def bi_linear_interpolation(image: npt.NDArray, p_i: float, p_j: float) -> float:
    """
    Function interpolates a pixel value at an unknown index ˙(p_i, p_j)˙.
    It takes in account the distances to all 4 neighbouring
    pixels and weights pixel contribution according to these distances.

    :param image: Input image.
    :param p_i: Index i
    :param p_j: Index j
    :return: Interpolated value.
    """
    # print("bi_linear_interpolation()")
    i, j = round(p_i), round(p_j)

    # Left, right, top, bottom pixels
    p_l = i, j - 1
    p_t = i - 1, j
    p_r = i, j + 1
    p_b = i + 1, j

    # distances to left, right, top, bottom pixels
    d_l = math.sqrt((p_i - p_l[0]) ** 2 + (p_j - p_l[1]) ** 2)
    d_t = math.sqrt((p_i - p_t[0]) ** 2 + (p_j - p_t[1]) ** 2)
    d_r = math.sqrt((p_i - p_r[0]) ** 2 + (p_j - p_r[1]) ** 2)
    d_b = math.sqrt((p_i - p_b[0]) ** 2 + (p_j - p_b[1]) ** 2)

    # Compute interpolated value
    inter_val = (d_r * get_val(image, p_l) + d_l * get_val(image, p_r)
                 + d_t * get_val(image, p_b) + d_b * get_val(image, p_t)) / (d_r + d_l + d_t + d_b)

    return inter_val


@njit
def multiscale_local_binary_pattern(image: npt.NDArray, num_points: int = 8, radius: int = 1) -> npt.NDArray:
    """
    Function computes local binary patterns for an arbitrary number of points and radius.

    :param image: Input image
    :param num_points: Number of points to consider around each pixel.
    :param radius: Raddi at which we sample points.
    :return: A 2 dimensional feature map.
    """

    height, width = image.shape
    image_lbp = np.zeros((height, width))
    d_theta = np.deg2rad(360 // num_points)
    OFF_GRID_THRESHOLD = 1e-5

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):

            # Compute decimal value of current pixel (i, j).
            code_ij = 0
            for p in range(num_points):
                p_i = i + (- radius * math.sin(p * d_theta))
                p_j = j + (radius * math.cos(p * d_theta))

                # Check if pixel is on the image
                if round(p_i) >= height or round(p_i) < 0 or round(p_j) >= width or round(p_j) < 0:
                    continue

                # If pixel doesn't fall exactly in the center, estimate its value by bi-linear interpolation.
                if abs(p_i - round(p_i)) >= OFF_GRID_THRESHOLD or abs(p_j - round(p_j)) >= OFF_GRID_THRESHOLD:
                    pixel_val = bi_linear_interpolation(image, p_i, p_j)
                else:
                    pixel_val = image[round(p_i)][round(p_j)]

                # Assign a value by comparing pixels.
                if pixel_val > image[i][j]:
                    pixel_val = 1
                else:
                    pixel_val = 0

                code_ij += pixel_val * (2 ** p)

            # Store code_ij at index i, j
            image_lbp[i][j] = code_ij

    return image_lbp


@njit
def uniform_local_binary_patterns(image: npt.NDArray, num_points: int = 8, radius: int = 1) -> npt.NDArray:
    """
    Function computes uniform local binary patterns for an arbitrary number of points and radius.

    :param image: Input image
    :param num_points: Number of points to consider around each pixel.
    :param radius: Raddi at which we sample points.
    :return: A 2 dimensional feature map.
    """

    height, width = image.shape
    image_lbp = np.zeros((height, width))
    d_theta = np.deg2rad(360 // num_points)
    OFF_GRID_THRESHOLD = 1e-5

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):

            # Compute decimal value of current pixel (i, j).
            code_ij = []
            for p in range(num_points):
                p_i = i + (- radius * math.sin(p * d_theta))
                p_j = j + (radius * math.cos(p * d_theta))

                # Check if pixel is on the image
                if round(p_i) >= height or round(p_i) < 0 or round(p_j) >= width or round(p_j) < 0:
                    continue

                # If pixel doesn't fall exactly in the center, estimate its value by bi-linear interpolation.
                if abs(p_i - round(p_i)) >= OFF_GRID_THRESHOLD or abs(p_j - round(p_j)) >= OFF_GRID_THRESHOLD:
                    pixel_val = bi_linear_interpolation(image, p_i, p_j)
                else:
                    pixel_val = image[round(p_i)][round(p_j)]

                # Assign a value by comparing pixels.
                if pixel_val > image[i][j]:
                    pixel_val = 1
                else:
                    pixel_val = 0

                code_ij.append(pixel_val)

            # Compute number of circular transitions
            code_ij = np.array(code_ij)
            U_lbp = abs(code_ij[-1] - code_ij[0]) \
                    + sum([abs(code_ij[i] - code_ij[i - 1]) for i in range(1, len(code_ij))])

            # Check if code is uniform or not and compute corresponding pixel value
            if U_lbp <= 2:
                code_ij = sum(code_ij)
            else:
                code_ij = num_points + 1

            image_lbp[i][j] = code_ij

    return image_lbp


class MultiscaleLBP:
    def __init__(self, num_points: int = 8, radius: int = 1, to_hist: bool = True, method: str = "default"):
        self.num_points = num_points
        self.radius = radius
        self.to_hist = to_hist
        self.method = method

    def describe(self, images: npt.NDArray, *args, **kwargs) -> npt.NDArray:
        """
        Function computes a feature vector for each image.

        :param images: A set of images.
        :return: A set of feature vectors.
        """
        fvs = []

        if self.to_hist:
            image_size_divisors = list(divisors(images[0].shape[0], generator=True))[1: -1]
            tile_width = kwargs.get("tile_width", image_size_divisors[len(image_size_divisors) // 2])
            print(f"tile width: {tile_width}")
            print(f"number of tiles: {(images[0].shape[0] / tile_width) ** 2}")

        if self.method == "default":
            # Extract features for each image
            for image in tqdm(images):
                image_lbp = multiscale_local_binary_pattern(image, self.num_points, self.radius)

                # Compute feature vector
                if self.to_hist:
                    fv = compute_concatenated_histogram(image_lbp, tile_size=(tile_width, tile_width),
                                                        num_bins=2 ** self.num_points)
                else:
                    fv = image_lbp.flatten()

                # Store feature vector
                fvs.append(fv)

            fvs = np.array(fvs)

        elif self.method == "uniform":
            for image in tqdm(images):
                image_lbp = uniform_local_binary_patterns(image, self.num_points, self.radius)

                # Compute feature vector
                if self.to_hist:
                    num_bins = int(image_lbp.max()) + 1
                    fv = compute_concatenated_histogram(image_lbp, tile_size=(tile_width, tile_width),
                                                        num_bins=num_bins)
                else:
                    fv = image_lbp.flatten()

                # Store feature vector
                fvs.append(fv)

        else:
            print(f"Specified method: {self.method} not supported !")

        return fvs

    def __str__(self):
        return "MultiscaleLBP"


if __name__ == "__main__":
    # Load data
    data_dir = "./data/awe/"
    results_dir = "./results/"
    IMG_SIZE = 128
    print("Loading data...")
    X, y = load_data(data_dir, IMG_SIZE, IMG_SIZE, normalize=False)
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")

    X = X[:1]
    y = y[:1]

    desc = ScikitLBP(num_points=8, radius=1, to_hist=False, method="uniform")
    X_lbp = desc.describe(X)

    X_lbp = np.array(X_lbp)
    X_lbp = X_lbp.reshape(1, IMG_SIZE, IMG_SIZE)
    plt.imshow(X_lbp[0], cmap="gray")
    plt.title("Scikit nri_uniform descriptor results.")
    plt.savefig("./img/scikit_nri_uniform")

    desc = MultiscaleLBP(num_points=8, radius=1, to_hist=False, method="uniform")
    X_lbp = desc.describe(X)
    X_lbp = np.array(X_lbp)
    X_lbp = X_lbp.reshape(1, IMG_SIZE, IMG_SIZE)
    plt.imshow(X_lbp[0], cmap="gray")
    plt.title("My nri_uniform descriptor results.")
    plt.savefig("./img/my_nri_uniform")

