import numpy as np

from .core import Diffable, Tensor


class Activation(Diffable):
    @property
    def weights(self):
        return []

    def get_weight_gradients(self):
        return []


################################################################################
## Intermediate Activations To Put Between Layers


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        return np.where(x > 0, x, self.alpha * x)

    def get_input_gradients(self) -> list[Tensor]:
        (x,) = self.inputs
        batch_size, features = x.shape
        slopes = np.where(x > 0, 1.0, self.alpha)
        jacobian = np.zeros((batch_size, features, features), dtype=x.dtype)
        for i in range(batch_size):
            np.fill_diagonal(jacobian[i], slopes[i])
        return [Tensor(jacobian)]


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0.0)


################################################################################
## Output Activations For Probability-Space Outputs


class Sigmoid(Activation):
    def forward(self, x) -> Tensor:
        return 1.0 / (1.0 + np.exp(-x))

    def get_input_gradients(self) -> list[Tensor]:
        (x,) = self.inputs
        batch_size, features = x.shape
        sig = 1.0 / (1.0 + np.exp(-x))
        slopes = sig * (1 - sig)
        jacobian = np.zeros((batch_size, features, features), dtype=x.dtype)
        for i in range(batch_size):
            np.fill_diagonal(jacobian[i], slopes[i])
        return [Tensor(jacobian)]


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def forward(self, x):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(x_shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def get_input_gradients(self):
        (x,) = self.inputs
        (y,) = self.outputs
        batch_size, features = x.shape
        grad = np.zeros((batch_size, features, features), dtype=x.dtype)
        for b in range(batch_size):
            y_b = y[b]
            outer = -np.outer(y_b, y_b)
            np.fill_diagonal(outer, y_b * (1 - y_b))
            grad[b] = outer
        return [Tensor(grad)]
