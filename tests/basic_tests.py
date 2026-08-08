import numpy as np
from mlp_sboa.model import forward_pass


def test_forward_shapes():
    X = np.random.randn(5, 3)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4, 1)
    b2 = np.random.randn(1)
    out = forward_pass({"W1": W1, "W2": W2}, {"b1": b1, "b2": b2}, X)
    assert out["A1"].shape == (5, 4)
    assert out["Y_hat"].shape == (5, 1)


if __name__ == '__main__':
    test_forward_shapes()
    print('basic tests passed')
