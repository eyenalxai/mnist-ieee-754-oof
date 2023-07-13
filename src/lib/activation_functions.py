import numpy as np


def relu_deriv(*, net_input: np.ndarray) -> np.ndarray:
    return net_input > 0  # type: ignore


def softmax(net_input: np.ndarray) -> np.ndarray:
    return np.exp(net_input) / sum(np.exp(net_input))


def relu(net_input: np.ndarray) -> np.ndarray:
    return np.maximum(net_input, 0)


def linear(net_input: np.ndarray) -> np.ndarray:
    return net_input


def ieee754_linear(net_input: np.ndarray) -> np.ndarray:
    return net_input + 1024.0 - 1024.0
