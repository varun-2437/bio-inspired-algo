import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_pass(weights, biases, X):
    """Forward pass for a single-hidden-layer MLP.

    weights: dict with 'W1' (input_dim x hidden_dim) and 'W2' (hidden_dim x output_dim)
    biases: dict with 'b1' (hidden_dim,) and 'b2' (output_dim,)
    X: numpy array (n_samples x input_dim)
    """
    W1 = weights["W1"]
    W2 = weights["W2"]
    b1 = biases["b1"]
    b2 = biases["b2"]

    Z1 = X.dot(W1) + b1.reshape((1, -1))
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2.reshape((1, -1))
    Y_hat = sigmoid(Z2)
    return {"A1": A1, "Y_hat": Y_hat}


def predict(weights, biases, X, data_y=None):
    """Predict and (optionally) un-normalize if data_y bounds provided.

    data_y: tuple (min, max) or None
    """
    out = forward_pass(weights, biases, X)
    yhat = out["Y_hat"]
    if data_y is not None:
        ymin, ymax = data_y
        yhat = yhat * (ymax - ymin) + ymin
    return yhat
