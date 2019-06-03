import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing

NUM_CLUSTER = 3
NUM_SAMPLES = 100
NUM_FEATURES = 2
MIN_VALUE = -2
MAX_VALUE = 2
MIN_STRATEGY = -1
MAX_STRATEGY = 1
IND_SIZE = 2 * NUM_CLUSTER

centers = [(-7, -4), (5, 1), (7, -4)]
X, y = make_blobs(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, cluster_std=1.0,
                  centers=centers, shuffle=True, random_state=0)


X = preprocessing.scale(X)
print(X)


def main():
    from es_handler import ES
    from rbf_handler import RBF

    # For 2 classes must became uncomment
    # for i in range(y.size):
    #     if y[i] == 0:
    #         y[i] = -1
    print(y)

    rbf = RBF(X, y)
    es = ES(rbf.eval_classification, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    bestGen = es.run_algorithm()
    best_individual = bestGen[0][0]
    print(best_individual)
    show_result(X, best_individual)


def show_result(X, v):
    x_data = X[:, 0]
    y_data = X[:, 1]
    v = np.reshape(v, (-1, 2))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.scatter(x_data, y_data, s=20)
    plt.scatter(v[:, 0], v[:, 1], s=300, marker='+', c="red")
    plt.show()


if __name__ == "__main__":
    main()
