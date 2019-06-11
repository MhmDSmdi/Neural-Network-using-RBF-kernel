import numpy as np


class RBF:
    W_matrix = V_matrix = radius = None
    LAMDA = 1

    def __init__(self, x, y, num_circles, is_regression):
        self.X = x
        self.y = y
        self.NUM_FEATURES = x.shape[1]
        self.NUM_CIRCLE = num_circles
        self.NUM_SAMPLES = x.shape[0]
        self.is_regression = is_regression

    def evaluate(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES + 1))
        self.radius = self.V_matrix[:, 0]
        self.V_matrix = self.V_matrix[:, 1:]
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CIRCLE))
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_CIRCLE):
                G_matrix[i, j] = self.cal_g(self.radius[j], self.X[i], self.V_matrix[j])
        self.W_matrix = self.cal_W(G_matrix, self.y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        if self.is_regression:
            loss = self.loss_regression(self.y, y_star)
        else:
            loss = self.loss_classification(self.y, y_star)
        return loss,

    def predict(self, individual):
        self.V_matrix = np.reshape(individual, (-1, self.NUM_FEATURES + 1))
        self.radius = self.V_matrix[:, 0]
        self.V_matrix = self.V_matrix[:, 1:]
        G_matrix = np.empty((self.NUM_SAMPLES, self.NUM_CIRCLE))
        for i in range(self.NUM_SAMPLES):
            for j in range(self.NUM_CIRCLE):
                G_matrix[i, j] = self.cal_g(self.radius[j], self.X[i], self.V_matrix[j])
        self.W_matrix = self.cal_W(G_matrix, self.y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        if self.is_regression:
            return y_star
        else:
            # y_star = 1 / (1 + np.e ** -y_star)
            return np.argmax(y_star, axis=1)

    def validation(self, X, y):
        G_matrix = np.empty((X.shape[0], self.NUM_CIRCLE))
        for i in range(X.shape[0]):
            for j in range(self.NUM_CIRCLE):
                G_matrix[i, j] = self.cal_g(self.radius[j], X[i], self.V_matrix[j])
        self.W_matrix = self.cal_W(G_matrix, y)
        y_star = np.matmul(G_matrix, self.W_matrix)
        if self.is_regression:
            return y_star, self.loss_regression(y, y_star)
        else:
            # y_star = 1 / (1 + np.e ** -y_star)
            return np.argmax(y_star, axis=1), self.loss_classification(y, y_star)

    def cal_W(self, G, y):
        regulated_G = np.matmul(np.transpose(G), G) + self.LAMDA * np.eye(self.NUM_CIRCLE)
        inverse = np.matmul(np.linalg.inv(regulated_G), np.transpose(G))
        return np.matmul(inverse, y)

    @staticmethod
    def cal_g(radius, x, v):
        vector = radius * np.dot(x-v, x-v)
        return np.exp(-vector)

    @staticmethod
    def loss_regression(y, y_star):
        return 0.5 * np.matmul((y_star - y).T, y_star - y)

    @staticmethod
    def loss_classification(y, y_star):
        return ((np.sum(np.sign(np.abs(np.argmax(y, axis=1) - np.argmax(y_star, axis=1))))) / len(y_star))
