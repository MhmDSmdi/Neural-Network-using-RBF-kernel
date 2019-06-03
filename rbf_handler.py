import math

import numpy as np


class RBF:
    W_matrix = V_matrix = None
    GAMA = 0.1
    LAMDA = 1

    def __init__(self, x, y):
        self.X = x
        self.y = y
        self.NUM_FEATURES = x.shape[1]
        self.NUM_CLUSTER = max(y) + 1
        self.NUM_SAMPLES = x.shape[0]

    def eval_classification(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CLUSTER))
        y_prime = np.zeros((self.NUM_SAMPLES, self.NUM_CLUSTER))
        for idx, label in enumerate(self.y):
            y_prime[idx][label] = 1
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_FEATURES):
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]), (self.X[i] - self.V_matrix[j])))
        self.W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)), np.transpose(G_matrix)), y_prime)
        y_star = np.matmul(G_matrix, self.W_matrix)
        loss = self.loss_classification(y_prime, y_star)
        # loss = self.loss_two_class(self.y, y_star)
        return loss,

    def eval_regression(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CLUSTER))
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_FEATURES):
                # dist = np.linalg.norm(self.X[i] - self.V_matrix[j]) ** 2
                # dist = self.GAMA * dist
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]), (self.X[i] - self.V_matrix[j])))
        # regulatedG = G_matrix.transpose().dot(G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)
        self.W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)),np.transpose(G_matrix)), self.y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        loss = self.loss_regression(self.y, y_star)
        return loss,

    def predict_classification(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CLUSTER))
        y_prime = np.zeros((self.NUM_SAMPLES, self.NUM_CLUSTER))
        for idx, label in enumerate(self.y):
            y_prime[idx][label] = 1
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_FEATURES):
                # dist = np.linalg.norm(self.X[i] - self.V_matrix[j]) ** 2
                # dist = self.GAMA * dist
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]),
                                                                   (self.X[i] - self.V_matrix[j])))
        # regulatedG = G_matrix.transpose().dot(G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)
        self.W_matrix = np.matmul(np.matmul(
            np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)),
            np.transpose(G_matrix)), y_prime)
        y_star = np.matmul(G_matrix, self.W_matrix)
        return np.argmax(y_star, axis=1)

    def predict_regression(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CLUSTER))
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_FEATURES):
                # dist = np.linalg.norm(self.X[i] - self.V_matrix[j]) ** 2
                # dist = self.GAMA * dist
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]),
                                                                   (self.X[i] - self.V_matrix[j])))
        # regulatedG = G_matrix.transpose().dot(G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)
        self.W_matrix = np.matmul(np.matmul(
            np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)),
            np.transpose(G_matrix)), self.y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        return y_star

    @staticmethod
    def loss_regression(y, y_star):
        return (1 / (2 * len(y))) * np.dot(np.transpose(y_star - y), (y_star - y))

    @staticmethod
    def loss_classification(y, y_star):
        return 1 - ((np.sum(np.sign(np.abs(np.argmax(y, axis=1) - np.argmax(y_star, axis=1))))) / len(y_star))

    @staticmethod
    def loss_two_class(y, y_star):
        return 0.5 * np.subtract(y, y_star).transpose().dot(np.subtract(y, y_star))
