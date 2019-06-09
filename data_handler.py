import numpy as np
import sklearn
from sklearn.datasets import make_blobs
import csv


def read_csv(file_name):
    data = []
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            data.append(list(map(float, row)))
    data = np.array(data)
    y = data[:, data.shape[1] - 1]
    X = data[:, :data.shape[1]]
    return X, y


def load_data_set(file_name, test_size):
    X, y = read_csv(file_name)
    X = sklearn.preprocessing.scale(X)
    num_class = max(y) + 1
    y = vector2_one_hot(y, X.shape[0], int(num_class))
    X_train = X[test_size:, :]
    X_test = X[0:test_size, :]
    y_train = y[test_size:]
    y_test = y[0:test_size]
    return X_train, X_test, y_train, y_test


def generate_multi_class_data_blobs(num_class, num_feature, num_data, test_size=10, min_val=-10, max_val=10):
    centers = []
    for i in range(num_class):
        xi = np.random.uniform(min_val, max_val, 1)
        yi = np.random.uniform(min_val, max_val, 1)
        centers.append((xi[0], yi[0]))
    X, y = make_blobs(n_samples=num_data + test_size, n_features=num_feature, cluster_std=1.0, centers=centers,
                      shuffle=True, random_state=1)
    X = sklearn.preprocessing.scale(X)
    # data = np.append(X, y, axis=1)
    y = vector2_one_hot(y, num_data + test_size, num_class)
    X_train = X[test_size:, :]
    X_test = X[0:test_size, :]
    y_train = y[test_size:]
    y_test = y[0:test_size]
    # print(data)
    return X_train, X_test, y_train, y_test


def generate_regression_date(num_feature, num_data, test_size):
    return


def vector2_one_hot(vec, row, column):
    y_prime = np.zeros((row, column))
    for idx, label in enumerate(vec):
        y_prime[idx][label] = 1
    return y_prime


if __name__ == '__main__':
    read_csv("data.csv")
