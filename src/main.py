from collections.abc import Callable

import numpy as np

from src.lib.activation_functions import linear, ieee754_linear, relu, softmax
from src.lib.data_processing import get_data
from src.lib.evaluation import predict, get_accuracy
from src.lib.propagation import output_only_forward_propagation
from src.lib.train import train_model


def test_model(
    *,
    data: np.ndarray,
    weights_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_output: np.ndarray,
    bias_output: np.ndarray,
    activation_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    output_activation = output_only_forward_propagation(
        weights_hidden=weights_hidden,
        bias_hidden=bias_hidden,
        weights_output=weights_output,
        bias_output=bias_output,
        data=data,
        activation_fn=activation_fn,
    )

    return predict(output_activation=output_activation)


def main() -> None:
    (
        train_data,
        train_labels,
        test_data,
        test_labels,
        m_train,
        total_samples,
    ) = get_data()

    activation_fns = [linear, ieee754_linear, relu]

    results = []
    for activation_fn in activation_fns:
        weights_hidden, bias_hidden, weights_output, bias_output = train_model(
            input_data=train_data,
            train_labels=train_labels,
            alpha=0.1,
            iterations=100,
            total_samples=total_samples,
            activation_fn=activation_fn,
        )

        dev_predictions = test_model(
            data=test_data,
            weights_hidden=weights_hidden,
            bias_hidden=bias_hidden,
            weights_output=weights_output,
            bias_output=bias_output,
            activation_fn=activation_fn,
        )

        accuracy = get_accuracy(predictions=dev_predictions, y=test_labels)
        results.append({"activation_fn": activation_fn.__name__, "accuracy": accuracy})

    print("\n\nresults:")
    for result in results:
        print(f"{result['activation_fn']}: {result['accuracy']}")


if __name__ == "__main__":
    main()
