import numpy as np
from mlp_sboa.model import forward_pass


def mse_loss(weights, biases, X, Y):
    out = forward_pass(weights, biases, X)
    Y_hat = out["Y_hat"]
    return float(np.mean((Y - Y_hat) ** 2))


def fitness_function(params, X_train, Y_train, input_dim, hidden_dim, output_dim):
    # Unpack parameters into W1, b1, W2, b2
    idx = 0
    W1_size = input_dim * hidden_dim
    W1 = params[idx: idx + W1_size].reshape((input_dim, hidden_dim))
    idx += W1_size

    b1 = params[idx: idx + hidden_dim]
    idx += hidden_dim

    W2_size = hidden_dim * output_dim
    W2 = params[idx: idx + W2_size].reshape((hidden_dim, output_dim))
    idx += W2_size

    b2 = params[idx: idx + output_dim]

    weights = {"W1": W1, "W2": W2}
    biases = {"b1": b1, "b2": b2}

    return mse_loss(weights, biases, X_train, Y_train)
