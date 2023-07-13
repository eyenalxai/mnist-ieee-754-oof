from collections.abc import Callable

import numpy as np

from src.lib.activation_functions import (
    relu_deriv,
    softmax,
    relu,
)


def one_hot_encode(*, arr: np.ndarray) -> np.ndarray:
    encoded_array = np.zeros((arr.max() + 1, arr.size))
    encoded_array[arr, np.arange(arr.size)] = 1
    return encoded_array


def output_only_forward_propagation(
    *,
    weights_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_output: np.ndarray,
    bias_output: np.ndarray,
    data: np.ndarray,
    activation_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    _, _, output_activation = forward_propagation(
        weights_hidden=weights_hidden,
        bias_hidden=bias_hidden,
        weights_output=weights_output,
        bias_output=bias_output,
        data=data,
        activation_fn=activation_fn,
    )
    return output_activation


def forward_propagation(
    *,
    weights_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_output: np.ndarray,
    bias_output: np.ndarray,
    data: np.ndarray,
    activation_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_layer_values = weights_hidden.dot(data) + bias_hidden
    hidden_activation = activation_fn(hidden_layer_values)

    output_layer_values = weights_output.dot(hidden_activation) + bias_output
    output_activation = activation_fn(output_layer_values)

    return hidden_layer_values, hidden_activation, output_activation


def backward_propagation(
    *,
    total_samples: int,
    hidden_layer_values: np.ndarray,
    hidden_activation: np.ndarray,
    output_activation: np.ndarray,
    weights_output: np.ndarray,
    input_data: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_hot_encoding_labels = one_hot_encode(arr=train_labels)
    delta_output_layer_values = output_activation - one_hot_encoding_labels

    delta_weights_output = 1 / total_samples * delta_output_layer_values.dot(hidden_activation.T)
    delta_bias_output = 1 / total_samples * np.sum(delta_output_layer_values)
    delta_hidden_layer_values = weights_output.T.dot(delta_output_layer_values) * relu_deriv(
        net_input=hidden_layer_values
    )

    delta_weights_hidden = 1 / total_samples * delta_hidden_layer_values.dot(input_data.T)
    delta_bias_hidden = 1 / total_samples * np.sum(delta_hidden_layer_values)

    return (
        delta_weights_hidden,
        delta_bias_hidden,
        delta_weights_output,
        delta_bias_output,
    )
