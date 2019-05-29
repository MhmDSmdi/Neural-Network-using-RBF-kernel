import random

import math as math
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

centers = [(-5, -5), (5, 5)]
X, y = make_blobs(n_samples=30, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=0)

for i in range(y.size):
    if y[i] == 0:
        y[i] = -1

# print(y)

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 4

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluateInd(individual):
    # Do some computation
    v = np.empty((2, 2))
    v[0] = individual[0:2]
    v[1] = individual[2:4]

    # data = X[0]
    # print(data)
    # for j in range(v.shape[0]):
    #     dist = np.linalg.norm(data-v[j]) ** 2
    #     dist = dist / 10
    #     print(dist)
    #     # G[i, j] = math.exp(dist)

    G = np.empty((30, 2))

    for i in range(X.shape[0]):
        data = X[i]
        for j in range(v.shape[0]):
            dist = np.linalg.norm(data - v[j]) ** 2
            dist = dist / 10
            G[i, j] = math.exp(-dist)

    # print(G.transpose())
    # print(G)
    g = G.transpose().dot(G)
    invg = np.linalg.inv(g)
    weights = (invg.dot(G.transpose())).dot(y)

    ylabel = G.dot(weights)

    # print(ylabel)
    sub = np.subtract(ylabel, y)
    err = sub.transpose().dot(sub)
    err = err / 2
    # print(err)

    # result = individual[0]
    return err,


# individual = [-5, -10, 4, 7]
# print(evaluateInd(individual))


# x = X[:, 0]
# y1 = X[:, 1]
# plt.plot(x, y1, "ob")
# plt.show()

# ind1 = toolbox.individual()
# print(ind1)
# ind1.fitness.values = evaluateInd(ind1)
# # print(ind1.fitness.valid)    # True
# print(ind1.fitness)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluateInd)

pop = toolbox.population(n=20)
bestGen = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100)
print(bestGen[0])
