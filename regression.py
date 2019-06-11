from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import data_handler
import numpy as np

X_train, X_test, y_train, y_test, X, y = data_handler.load_data_excel("./dataset/regdata2000.xlsx", 800, num_class=0,
                                                                is_regression=True, shuffle=False, preprocessing=True)

NUM_CIRCLE = 5
# NUM_SAMPLES = 150
NUM_FEATURES = 3
MIN_VALUE = np.min(X_train)
MAX_VALUE = np.max(X_train)
MIN_STRATEGY = -5
MAX_STRATEGY = 5
IND_SIZE = (NUM_FEATURES + 1) * NUM_CIRCLE



def main():
    from es_handler import ES
    from rbf_handler import RBF
    rbf = RBF(X_train, y_train, NUM_CIRCLE, is_regression=True)
    es = ES(rbf.evaluate, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    pop, hof, err_train = es.run_algorithm(ngen=10)
    y_hat, err_test = rbf.validation(X_test, y_test)
    print(err_train)
    print(err_test)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    show_result(X, y, 'o', ax, c='b', start_point=0)
    show_result(X_test, y_hat, '+', ax, c='r', start_point=0)
    plt.show()


def show_result(X, y, marker, ax, c, start_point=0):
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    n = np.arange(start_point, X.shape[0] + start_point)
    ax.scatter(n, y, c=c, marker=marker)


if __name__ == "__main__":
    main()
