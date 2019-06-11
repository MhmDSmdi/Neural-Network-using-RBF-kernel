from matplotlib import pyplot as plt
import numpy as np
from sympy.physics.quantum.tests.test_circuitplot import mpl
import sklearn

import data_handler
from es_handler import ES
from rbf_handler import RBF


def show_input(X):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


NUM_CLASS = 2
# X_train, X_test, y_train, y_test = data_handler.load_data_csv("./dataset/5clstest5000.csv", 1000, NUM_CLASS, is_regression=False, preprocessing=True, shuffle=True)
X_train, X_test, y_train, y_test, X, y = data_handler.load_data_excel("./dataset/2clstrain5000.xlsx", 1000, NUM_CLASS, is_regression=False, preprocessing=True, shuffle=True)

NUM_SAMPLES = X_train.shape[0]
NUM_FEATURES = X_train.shape[1]
NUM_CIRCLE = 10
MIN_VALUE = np.min(X_train)
MAX_VALUE = np.max(X_train)
MIN_STRATEGY = -5
MAX_STRATEGY = 5
IND_SIZE = (NUM_FEATURES + 1) * NUM_CIRCLE


def main():
    s_train = "Train Accuracy = {} \n (num_train={}, num_circle={})"
    s_test = "Test Accuracy = {} \n (num_test={}, num_circle={})"
    rbf = RBF(X_train, y_train, NUM_CIRCLE, False)
    es = ES(rbf.evaluate, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    pop, hof, err_train = es.run_algorithm(ngen=10)
    y_hat = rbf.predict(hof[0])
    y_validation, err_test = rbf.validation(X_test, y_test)
    print("Train Accuracy = {} ".format(1 - err_train))
    print("Test Accuracy = {}".format(1 - err_test))
    show_result(X_train, y_hat, rbf, s_train.format(1 - err_train, X_train.shape[0], NUM_CIRCLE))
    show_result(X_test, y_validation, rbf, s_test.format(1 - err_test, X_test.shape[0], NUM_CIRCLE))


def show_result(X, y, rbf, title):
    x_data = X[:, 0]
    y_data = X[:, 1]
    # n = np.max(y) + 1
    n = NUM_CLASS
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, n, n + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scat = ax.scatter(x_data, y_data, s=50, c=y, cmap=cmap, norm=norm)
    plt.title(title)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Clusters')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    # plt.scatter(rbf.V_matrix[:, 0], rbf.V_matrix[:, 1], s=100, c='red', marker='+')
    plt.show()


if __name__ == "__main__":
    main()
