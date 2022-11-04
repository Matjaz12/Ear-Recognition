import numpy as np
from sympy import divisors
from tqdm import tqdm
import numpy.typing as npt

from helpers import compute_concatenated_histogram


def simple_local_binary_pattern(image: npt.NDArray) -> npt.NDArray:
    """
    Function computes local binary pattern, assuming the following
    (number of points: 8, radius: 1).

    :param image: Input image
    :return: 2 dimensional feature map.
    """
    height, width = image.shape
    X_lbp = np.zeros((height, width))

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Compute decimal value of current pixel (i, j).
            code_ij, n = 0, 0

            # Iterate over the 8 neighbouring points
            for k in range(-1, 2):
                for g in range(-1, 2):
                    if k == 0 and g == 0:
                        continue

                    # Check if new point is on the image
                    elif i + k >= height or i + k < 0 or j + g >= width or j + g < 0:
                        continue

                    # Assign a value by comparing pixels.
                    val = 0
                    if image[i + k][j + g] >= image[i][j]:
                        val = 1

                    code_ij += val * (2 ** n)
                    n += 1
            X_lbp[i][j] = code_ij

    return X_lbp


class BasicLBP:
    def __init__(self, to_hist: bool = True):
        self.num_points = 8
        self.radius = 1
        self.to_hist = to_hist

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

        # Extract features for each image
        for image in tqdm(images):
            image_lbp = simple_local_binary_pattern(image)

            # Compute feature vector
            if self.to_hist:
                fv = compute_concatenated_histogram(image_lbp, tile_size=(tile_width, tile_width),
                                                    num_bins=2 ** self.num_points)
            else:
                fv = image_lbp.flatten()

            # Store feature vector
            fvs.append(fv)

        fvs = np.array(fvs)
        return fvs

    def __str__(self):
        return "BasicLBP"
