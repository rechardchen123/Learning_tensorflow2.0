import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
'''
Let's start by loading and preparing the California housing dataset. We first load it, then split it into a training set, a validation set and a test set, and finally we scale it:
'''
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
'''
For very large datasets that do not fit in memory, you will typically want to split it into many files first, then have TensorFlow read these files in parallel. To demonstrate this, let's start by spliting the scaled housing dataset and saving it to 20 CSV files.
'''


def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filenames = []
    m = len(data)
    for file_idx, row_indices in enumerate(
            np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, "wt", encoding='utf-8') as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filenames


train_data = np.c_[X_train_scaled, y_train]
valid_data = np.c_[X_valid_scaled, y_valid]
test_data = np.c_[X_test_scaled, y_test]
header_cols = ["Scaled" + name
               for name in housing.feature_names] + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filenames = save_to_multiple_csv_files(
    train_data, "train_data", header, n_parts=20)
valid_filenames = save_to_multiple_csv_files(
    valid_data, "valid", header, n_parts=10)
test_filenames = save_to_multiple_csv_files(
    test_data, "test", header, n_parts=10)
