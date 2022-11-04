import os
import re

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from matplotlib import pyplot as plt


def load_data(data_path: str, new_width: int = -1, new_height: int = -1, normalize: bool = False) -> npt.NDArray:
    """
    Function iterates over the data directory and loads images and their corresponding labels
    into memory.

    @param data_path: Root directory where the data is located.
    @param new_width: Width of the read image.
    @param new_height: Height of the read image.
    @param normalize: Normalize pixel values or not.
    @return: A list of images `X` and a list of labels `y`.
    """
    X, y = [], []

    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(subdir, file)
            if re.match(r"^.*\.(png)", filepath):
                # Read and preprocess image
                """
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, dsize=(new_width, new_height))
                """

                image_data = Image.open(filepath).convert("L")
                image_data = image_data.resize((new_width, new_height))
                image = np.array(image_data)

                # Read the image label
                label = int(subdir.split("/")[-1])

                # Store the sample
                X.append(image)
                y.append(label)

    # Convert to nunpy array.
    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":
    X, y = load_data("data/awe", 128, 128)
    print(X.shape)
    print(y.shape)

    from collections import Counter
    labels_hist = Counter(y)
    plt.bar(labels_hist.keys(), labels_hist.values())
    plt.xlabel("Classes")
    plt.ylabel("Counter")
    plt.title("Distribution of classes")
    plt.show()
