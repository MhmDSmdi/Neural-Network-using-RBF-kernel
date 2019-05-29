import array
import math
import random

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

NUM_CLUSTER = 2
NUM_SAMPLES = 30
NUM_FEATURES = 2
IND_SIZE = 2 * NUM_CLUSTER
GAMA = 2

centers = [(-5, -5), (0, 5)]
X, y = make_blobs(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, cluster_std=1.0,
                  centers=centers, shuffle=True, random_state=0)

for i in range(y.size):
    if y[i] == 0:
        y[i] = -1

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMin)


def evaluate(individual):
    v = np.reshape(individual, (-1, NUM_FEATURES))
    G_matrix = np.empty((NUM_SAMPLES, NUM_FEATURES))
    for i in range(X.shape[0]):
        for j in range(v.shape[0]):
            dist = np.linalg.norm(X[i] - v[j]) ** 2
            dist = dist / 10
            G_matrix[i, j] = math.exp(-dist)
    W_matrix = (np.linalg.inv(G_matrix.transpose().dot(G_matrix)).dot(G_matrix.transpose())).dot(y)
    y_star = G_matrix.dot(W_matrix)
    loss = 0.5 * np.subtract(y_star, y).transpose().dot(np.subtract(y_star, y))
    return loss,


toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def main():
    MU = 20
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop = toolbox.population(n=MU)
    bestGen = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats)
    v = bestGen[0][0]
    show_result(X, v)


def show_result(X, v):
    x_data = X[:, 0]
    y_data = X[:, 1]
    v = np.reshape(v, (-1, 2))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    scat = ax.scatter(x_data, y_data, s=30)
    plt.scatter(v[:, 0], v[:, 1], s=400, marker='+', c="red")
    plt.show()



if __name__ == "__main__":
    main()
