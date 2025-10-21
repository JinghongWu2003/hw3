import numpy as np

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return Tensor(np.mean(np.mean(diff ** 2, axis=1)))


    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        batch_size, output_size = y_pred.shape
        scale = 2.0 / (batch_size * output_size)
        diff = y_pred - y_true

        grad_values = scale * diff
        grad_y_pred = np.zeros((batch_size, output_size, output_size), dtype=y_pred.dtype)
        grad_y_true = np.zeros_like(grad_y_pred)

        for i in range(batch_size):
            np.fill_diagonal(grad_y_pred[i], grad_values[i])
            np.fill_diagonal(grad_y_true[i], -grad_values[i])

        return [Tensor(grad_y_pred), Tensor(grad_y_true)]

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        per_sample_loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        loss = np.mean(per_sample_loss)
        return Tensor(loss)
    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        batch_size = y_pred.shape[0]
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        grad_values = -y_true / (y_pred_clipped * batch_size)

        num_classes = y_pred.shape[1]
        grad_y_pred = np.zeros((batch_size, num_classes, num_classes), dtype=y_pred.dtype)
        grad_y_true = np.zeros_like(grad_y_pred)

        for i in range(batch_size):
            np.fill_diagonal(grad_y_pred[i], grad_values[i])

        return [Tensor(grad_y_pred), Tensor(grad_y_true)]

