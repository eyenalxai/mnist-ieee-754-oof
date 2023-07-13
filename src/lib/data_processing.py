import numpy as np
import pandas as pd


def get_data():
    df = np.array(pd.read_csv("data/train.csv"))
    total_samples = df.shape[0]

    np.random.shuffle(df)

    test_data, train_data = np.split(df, [1000])
    test_labels, train_labels = np.split(np.array(df[:, 0]), [1000])
    test_data, train_data = (
        test_data[:, 1:].T / 255.0,
        train_data[:, 1:].T / 255.0,
    )

    return (
        train_data,
        train_labels,
        test_data,
        test_labels,
        train_data.shape[1],
        total_samples,
    )
