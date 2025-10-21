from collections import defaultdict
import numpy as np


class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, trainable_params, grads):
        for w, g in zip(trainable_params, grads):
            w.assign(w - self.learning_rate * g)


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        for i, (w, g) in enumerate(zip(trainable_params, grads)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (g ** 2)
            update = self.learning_rate * g / (np.sqrt(self.v[i]) + self.epsilon)
            w.assign(w - update)


class Adam:
    def __init__(
            self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)  # First moment zero vector
        self.v = defaultdict(lambda: 0)  # Second moment zero vector.
        self.t = 0  # Time counter

    def apply_gradients(self, trainable_params, grads):
        self.t += 1
        for i, (w, g) in enumerate(zip(trainable_params, grads)):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * g
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            w.assign(w - update)
