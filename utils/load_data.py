import pandas as pd
import numpy as np
import csv


def load_data(train_file, test_file, split_ratio=0.8, sep=',', header=None):
    train = pd.read_csv(train_file, sep=sep, header=0, quoting=1).sample(frac=1).values
    test = pd.read_csv(test_file, sep=sep, header=header, quoting=1).values

    split_part = int(len(train) * split_ratio)
    valid = train[split_part:]
    train = train[:split_part]

    return train[:, 1], train[:, 2], valid[:, 1], valid[:, 2], test[:, 1], test[:, 2]


def open_file(file_name):
    with open(file_name) as data_file:
        file = list(csv.reader(data_file, delimiter=','))
        items = np.array(file)
        return items
