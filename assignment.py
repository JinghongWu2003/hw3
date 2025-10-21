from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
            Dense(784, 10),
            Softmax()
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    return Adam(learning_rate=0.001)

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    return CategoricalCrossEntropy()

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return CategoricalAccuracy()

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    model = get_model()
    # 2. Compile the model with optimizer, loss function, and accuracy metric
    model.compile(
        optimizer=get_optimizer(),
        loss_fn=get_loss_fn(),
        acc_fn=get_acc_fn()
    )
    # 3. Load and preprocess the data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    encoder = OneHotEncoder()
    encoder.fit(y_train)
    y_train = encoder.forward(y_train)
    y_test = encoder.forward(y_test)
    print("x_train shape:", x_train.shape, "min/max:", float(x_train.min()), float(x_train.max()))
    print("y_train shape:", y_train.shape, "row sums (first 5):", y_train[:5].sum(axis=1))
    print("y_train unique values:", np.unique(y_train[:5]))
    assert x_train.ndim == 2 and x_train.shape[1] == 784, "输入必须展平为 (N, 784)"
    assert np.allclose(y_train.sum(axis=1), 1), "y_train 必须是 one-hot（每行和=1）"
    # 4. Train the model
    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=64
    )
    # 5. Evaluate the model
    print("\nFinal Evaluation:")
    model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=64
    )