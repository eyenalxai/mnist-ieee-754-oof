import numpy as np


def predict(*, output_activation: np.ndarray) -> np.ndarray:
    return np.argmax(output_activation, 0)


def get_accuracy(*, predictions: np.ndarray, y: np.ndarray) -> float:
    return np.sum(predictions == y) / y.size
