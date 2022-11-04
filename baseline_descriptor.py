import numpy.typing as npt


class BaselineDescriptor:
    @staticmethod
    def describe(X: npt.NDArray, *args, **kwargs) -> npt.NDArray:
        """
        Function flattens an input array of shape
        `[num samples, height, width] to an array of shape
        `[num samples, height * width]`.

        :param X: Input data.
        :return: Flattened data.
        """
        print("Extracting features...")
        X_features = X.reshape(X.shape[0], -1)
        return X_features
