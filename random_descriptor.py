import numpy as np
import numpy.typing as npt


class RandomDescriptor:
    @staticmethod
    def describe(X: npt.NDArray, *args, **kwargs) -> npt.NDArray:
        """
        Function returns a flattened vector containing random values
        for each element in the array X.

        :param X: Input data.
        :return: Set of random feature vectors.
        """

        print("Extracting features...")
        X_features = np.random.rand(X.shape[0], X.shape[1] * X.shape[2])

        return X_features
