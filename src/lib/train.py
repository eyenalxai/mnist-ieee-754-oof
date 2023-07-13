from collections.abc import Callable

import numpy as np

from src.lib.evaluation import predict, get_accuracy
from src.lib.propagation import backward_propagation, forward_propagation


def initialize_weights(method: str = "glorot") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method == "he":
        weights_hidden = np.random.randn(10, 784) * np.sqrt(2 / 784)
        weights_output = np.random.randn(10, 10) * np.sqrt(2 / 10)
    elif method == "glorot":
        weights_hidden = np.random.randn(10, 784) * np.sqrt(1 / 784)
        weights_output = np.random.randn(10, 10) * np.sqrt(1 / 10)
    else:
        raise ValueError("Invalid method: choose either 'he' or 'glorot'")

    bias_hidden = np.zeros((10, 1))
    bias_output = np.zeros((10, 1))

    return weights_hidden, bias_hidden, weights_output, bias_output


def update_params(
    *,
    weights_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_output: np.ndarray,
    bias_output: np.ndarray,
    delta_weights_hidden: np.ndarray,
    delta_bias_hidden: np.ndarray,
    delta_weights_output: np.ndarray,
    delta_bias_output: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights_hidden = weights_hidden - alpha * delta_weights_hidden
    bias_hidden = bias_hidden - alpha * delta_bias_hidden
    weights_output = weights_output - alpha * delta_weights_output
    bias_output = bias_output - alpha * delta_bias_output

    return weights_hidden, bias_hidden, weights_output, bias_output


def train_model(
    *,
    input_data: np.ndarray,
    train_labels: np.ndarray,
    alpha: float,
    iterations: int,
    total_samples: int,
    activation_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights_hidden, bias_hidden, weights_output, bias_output = initialize_weights()
    for i in range(iterations):
        hidden_layer_values, hidden_activation, output_activation = forward_propagation(
            weights_hidden=weights_hidden,
            bias_hidden=bias_hidden,
            weights_output=weights_output,
            bias_output=bias_output,
            data=input_data,
            activation_fn=activation_fn,
        )

        (
            delta_weights_hidden,
            delta_bias_hidden,
            delta_weights_output,
            delta_bias_output,
        ) = backward_propagation(
            hidden_layer_values=hidden_layer_values,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            weights_output=weights_output,
            input_data=input_data,
            train_labels=train_labels,
            total_samples=total_samples,
        )

        weights_hidden, bias_hidden, weights_output, bias_output = update_params(
            weights_hidden=weights_hidden,
            bias_hidden=bias_hidden,
            weights_output=weights_output,
            bias_output=bias_output,
            delta_weights_hidden=delta_weights_hidden,
            delta_bias_hidden=delta_bias_hidden,
            delta_weights_output=delta_weights_output,
            delta_bias_output=delta_bias_output,
            alpha=alpha,
        )

        if (i + 1) % 10 == 0:
            predictions = predict(output_activation=output_activation)
            accuracy = get_accuracy(predictions=predictions, y=train_labels)
            print(f"accuracy {round(accuracy, 2):<15} on iteration {i + 1:<15} for {activation_fn.__name__}")

    return weights_hidden, bias_hidden, weights_output, bias_output
