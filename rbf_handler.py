import math

import numpy as np


class RBF:
    W_matrix = V_matrix = None
    GAMA = 4
    LAMDA = 1

    def __init__(self, x, y, num_features, num_circles, num_samples):
        self.X = x
        self.y = y
        self.NUM_FEATURES = num_features
        self.NUM_CIRCLE = num_circles
        self.NUM_SAMPLES = num_samples

    def eval_classification(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CIRCLE))
        y_prime = np.zeros((self.NUM_SAMPLES, self.NUM_CIRCLE))
        for idx, label in enumerate(self.y):
            y_prime[idx][label] = 1
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_FEATURES):
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]), (self.X[i] - self.V_matrix[j])))
        self.W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CIRCLE)), np.transpose(G_matrix)), y_prime)
        y_star = np.matmul(G_matrix, self.W_matrix)
        loss = self.loss_classification(y_prime, y_star)
        # loss = self.loss_two_class(self.y, y_star)
        return loss,

    def eval_regression(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        # radius = self.V_matrix[:, 0]
        # self.V_matrix = self.V_matrix[:, 1:]
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CIRCLE))
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_CIRCLE):
                # dist = np.linalg.norm(self.X[i] - self.V_matrix[j]) ** 2
                # dist = self.GAMA * dist
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]), (self.X[i] - self.V_matrix[j])))
        # regulatedG = G_matrix.transpose().dot(G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)
        self.W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CIRCLE)), np.transpose(G_matrix)), self.y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        loss = self.loss_regression(self.y, y_star)
        return loss,

    def predict_classification(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CIRCLE))
        y_prime = np.zeros((self.NUM_SAMPLES, self.NUM_CIRCLE))
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
            np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CIRCLE)),
            np.transpose(G_matrix)), y_prime)
        y_star = np.matmul(G_matrix, self.W_matrix)
        return np.argmax(y_star, axis=1)

    def predict_regression(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES))
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CIRCLE))
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_FEATURES):
                # dist = np.linalg.norm(self.X[i] - self.V_matrix[j]) ** 2
                # dist = self.GAMA * dist
                G_matrix[i, j] = np.exp(-1 * self.GAMA * np.matmul(np.transpose(self.X[i] - self.V_matrix[j]),
                                                                   (self.X[i] - self.V_matrix[j])))
        # regulatedG = G_matrix.transpose().dot(G_matrix) + self.LAMDA * np.eye(self.NUM_CLUSTER)
        self.W_matrix = np.matmul(np.matmul(
            np.linalg.inv(np.matmul(np.transpose(G_matrix), G_matrix) + self.LAMDA * np.eye(self.NUM_CIRCLE)),
            np.transpose(G_matrix)), self.y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        return y_star

    @staticmethod
    def loss_regression(y, y_star):
            # print('error ', 0.5 * np.matmul((self.y_hat - self.y).T, self.y_hat - self.y))
        return 0.5 * np.matmul((y_star - y).T, y_star -y)
        # return (1 / (2 * len(y))) * np.dot(np.transpose(y_star - y), (y_star - y))

    @staticmethod
    def loss_classification(y, y_star):
        return 1 - ((np.sum(np.sign(np.abs(np.argmax(y, axis=1) - np.argmax(y_star, axis=1))))) / len(y_star))

    @staticmethod
    def loss_two_class(y, y_star):
        return 0.5 * np.subtract(y, y_star).transpose().dot(np.subtract(y, y_star))
