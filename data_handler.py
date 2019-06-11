import numpy as np
import sklearn
from sklearn.datasets import make_blobs
import csv
import xlrd


def read_csv(file_name, shuffel):
    data = []
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            data.append(list(map(float, row)))
    data = np.array(data)
    if shuffel:
        np.random.shuffle(data)
    np.random.shuffle(data)
    y = data[:, data.shape[1] - 1]
    X = data[:, :data.shape[1] - 1]
    return X, y


def load_data_csv(file_name, test_size, num_class, is_regression, shuffle=False, preprocessing=False):
    X, y = read_csv(file_name, shuffle)
    if preprocessing:
        X = sklearn.preprocessing.scale(X)
        if is_regression:
            y = sklearn.preprocessing.scale(y)
    if not is_regression:
        y = vector2_one_hot(y, X.shape[0], num_class)
    X_train = X[test_size:, :]
    X_test = X[0:test_size, :]
    y_train = y[test_size:]
    y_test = y[0:test_size]
    return X_train, X_test, y_train, y_test, X, y


def generate_multi_class_data_blobs(num_class, num_feature, num_data, test_size=10, min_val=-10, max_val=10):
    centers = []
    for i in range(num_class):
        xi = np.random.uniform(min_val, max_val, 1)
        yi = np.random.uniform(min_val, max_val, 1)
        centers.append((xi[0], yi[0]))
    X, y = make_blobs(n_samples=num_data + test_size, n_features=num_feature, cluster_std=1.0, centers=centers,
                      shuffle=True, random_state=1)
    X = sklearn.preprocessing.scale(X)
    y = vector2_one_hot(y, num_data + test_size, num_class)
    X_train = X[test_size:, :]
    X_test = X[0:test_size, :]
    y_train = y[test_size:]
    y_test = y[0:test_size]
    return X_train, X_test, y_train, y_test


def generate_regression_date(fun, num_feature, num_data, test_size, min, max):
    X = np.random.uniform(min, max, (num_data + test_size, num_feature))
    y = fun(X)
    X = sklearn.preprocessing.scale(X)
    y = sklearn.preprocessing.scale(y)
    X_test = X[:test_size, :]
    y_test = y[:test_size]
    X_train = X[test_size:, :]
    y_train = y[test_size:]
    return X_train, X_test, y_train, y_test


def load_data_excel(file_name, test_size, num_class, is_regression, shuffle=False, preprocessing=False):
    loc = (file_name)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    data = np.zeros((sheet.nrows, sheet.ncols))
    for i in range(sheet.nrows):
        for j in range(sheet.ncols):
            data[i, j] = sheet.cell_value(i, j)
    data = np.array(data)
    if shuffle:
        np.random.shuffle(data)
    y = data[:, data.shape[1] - 1]
    if not is_regression:
        y = vector2_one_hot(y, y.shape[0], num_class)
    data = data[:, :data.shape[1] - 1]
    if preprocessing:
        data = sklearn.preprocessing.scale(data)
        if is_regression:
            y = sklearn.preprocessing.scale(y)
    X_train = data[test_size:, :]
    X_test = data[0:test_size, :]
    y_train = y[test_size:]
    y_test = y[0:test_size]
    return X_train, X_test, y_train, y_test, data, y


def vector2_one_hot(vec, row, column):
    y_prime = np.zeros((row, column))
    for idx, label in enumerate(vec):
        y_prime[idx][int(label)] = 1
    return y_prime


if __name__ == '__main__':
    print((load_data_excel("2clstrain5000.xlsx", 10, False))[3].shape)
