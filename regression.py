import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

NUM_CIRCLE = 15
NUM_SAMPLES = 150
NUM_FEATURES = 1

MIN_VALUE = -5
MAX_VALUE = 5
MIN_STRATEGY = -3
MAX_STRATEGY = 3
# IND_SIZE = NUM_FEATURES * NUM_CIRCLE
IND_SIZE = (NUM_FEATURES + 1) * NUM_CIRCLE

# X, y = make_regression(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, noise=0.1)


X = np.random.uniform(-4*np.pi, 4*np.pi, (NUM_SAMPLES, NUM_FEATURES))
y = np.sin(X)/(np.pi*X)
# y = np.cos(X)


X = preprocessing.scale(X)
y = preprocessing.scale(y)


def main():
    from es_handler import ES
    from rbf_handler import RBF
    rbf = RBF(X, y, NUM_FEATURES, NUM_CIRCLE, NUM_SAMPLES, True)
    es = ES(rbf.evaluate, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    pop, hof = es.run_algorithm()
    y_hat = rbf.predict(hof[0])
    show_result(X, y, y_hat)
    # show_result(X, y, None)


def show_result(X, y, y_hat):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.scatter(X, y, s=10)
    ax.scatter(X, y_hat, c='red', s=10)
    plt.show()


if __name__ == "__main__":
    main()
