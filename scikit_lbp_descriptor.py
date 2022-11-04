import numpy as np
import numpy.typing as npt
from numba import njit
from skimage import feature
from tqdm import tqdm
from sympy import divisors

from helpers import compute_concatenated_histogram


class ScikitLBP:
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

        # Extract features for each image
        for image in tqdm(images):
            image_lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method=self.method)

            # Compute feature vector
            if self.to_hist:
                num_bins = int(image_lbp.max()) + 1
                fv = compute_concatenated_histogram(image_lbp, tile_size=(tile_width, tile_width),
                                                    num_bins=num_bins)
            else:
                fv = image_lbp.flatten()

            # Store feature vector
            fvs.append(fv)

        fvs = np.array(fvs)
        return fvs

    def __str__(self):
        return "ScikitLBP"
