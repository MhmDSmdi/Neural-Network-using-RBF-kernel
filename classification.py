import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing
from sympy.physics.quantum.tests.test_circuitplot import mpl

NUM_CIRCLE = 25
NUM_CLASSES = None
NUM_SAMPLES = 100
NUM_FEATURES = 2
MIN_VALUE = -2
MAX_VALUE = 2
MIN_STRATEGY = -1
MAX_STRATEGY = 1
IND_SIZE = (NUM_FEATURES + 1) * NUM_CIRCLE

centers = [(-7, -4), (5, 1), (7, -4), (0, 0)]
X, y = make_blobs(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, cluster_std=1.0,
                  centers=centers, shuffle=True, random_state=0)
X = preprocessing.scale(X)
NUM_CLASSES = np.max(np.ravel(y)) + 1
y_prime = np.zeros((NUM_SAMPLES, NUM_CLASSES))
for idx, label in enumerate(y):
    y_prime[idx][label] = 1


def main():
    from es_handler import ES
    from rbf_handler import RBF

    rbf = RBF(X, y_prime, NUM_FEATURES, NUM_CIRCLE, NUM_SAMPLES, False)
    es = ES(rbf.evaluate, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    pop, hof = es.run_algorithm(ngen=5)
    y_hat = rbf.predict(hof[0])
    print(y_hat)
    show_input(X)
    show_result(X, y_hat)


def show_result(X, y_hat):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    colors = ['r', 'y', 'g', 'black']
    for i in range(NUM_SAMPLES):
        plt.scatter(X[i, 0], X[i, 1], c=colors[y_hat[i]])
    plt.show()


def show_input(X):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
