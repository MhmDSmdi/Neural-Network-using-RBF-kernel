import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing
from sympy.physics.quantum.tests.test_circuitplot import mpl
import data_handler
from es_handler import ES
from rbf_handler import RBF

NUM_CIRCLE = 25
NUM_CLASSES = None
NUM_SAMPLES = 100
NUM_FEATURES = 2
MIN_VALUE = -2
MAX_VALUE = 2
MIN_STRATEGY = -1
MAX_STRATEGY = 1
IND_SIZE = (NUM_FEATURES + 1) * NUM_CIRCLE

# X_train, X_test, y_train, y_test, = data_handler.generate_multi_class_data_blobs(4, NUM_FEATURES, NUM_SAMPLES, test_size=40)
X_train, X_test, y_train, y_test, = data_handler.load_data_set("data.csv", 1)


def main():
    rbf = RBF(X_train, y_train, NUM_FEATURES, NUM_CIRCLE, NUM_SAMPLES, False)
    es = ES(rbf.evaluate, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    pop, hof = es.run_algorithm(ngen=5)
    y_hat = rbf.predict(hof[0])
    show_input(X_train, "Train : Input")
    show_result(X_train, y_hat, "Train : Output")
    y_validation, err = rbf.validation(X_test, y_test)
    show_input(X_test, "Test : Input")
    show_result(X_test, y_validation, "Test, Output")
    print(y_validation)
    print(err)


def show_result(X, y_hat, s):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    colors = ['r', 'y', 'g', 'black']
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], c=colors[y_hat[i]])
    plt.title(s)
    plt.show()


def show_input(X, s):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(s)
    plt.show()


if __name__ == "__main__":
    main()
