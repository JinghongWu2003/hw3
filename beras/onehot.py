import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def __init__(self):
        self.encoding_dict = None

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique= np.unique(data)
        onehot = np.eye(len(unique))
        self.encoding_dict={ label: onehot[i] for i, label in enumerate(unique)}

    def forward(self, data):
        return np.array([self.encoding_dict[label] for label in data])

    def inverse(self, data):
        inverse_dict = {i: label for label, i in zip(self.encoding_dict.keys(), range(len(self.encoding_dict)))}
        indices = np.argmax(data, axis=1)
        return np.array([inverse_dict[idx] for idx in indices])

