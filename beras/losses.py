import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        loss = np.mean(np.mean(diff ** 2, axis=1))
        return Tensor(loss)


    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        batch_size, output_size = y_pred.shape
        grad_y_pred = 2*(y_pred - y_true)/(batch_size*output_size)
        grad_y_true = -2*(y_pred - y_true)/(batch_size*output_size)
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
        grad_y_pred = -y_true / (y_pred_clipped * batch_size)
        grad_y_true = np.zeros_like(y_true)
        return [Tensor(grad_y_pred), Tensor(grad_y_true)]

