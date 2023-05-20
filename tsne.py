import numpy as np


class tSNE:
    def __init__(self, n_components, perplexity=30, learning_rate=200, num_iterations=1000, momentum=0.9):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.momentum = momentum

    def fit_transform(self, X):
        P = self.compute_pairwise_affinities(X)
        Y = np.random.randn(X.shape[0], self.n_components)

        for iteration in range(self.num_iterations):
            Q = self.compute_low_dimensional_affinities(Y)

            dY = self.compute_gradient(P, Q, Y)

            if iteration > 0:
                update = self.learning_rate * dY + self.momentum * update
            else:
                update = self.learning_rate * dY

            Y -= update

        return Y

    def compute_pairwise_affinities(self, X):
        pairwise_distances = self.compute_pairwise_distances(X)

        P = np.zeros((X.shape[0], X.shape[0]))
        sigma = np.sqrt(2 * self.perplexity)
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i != j:
                    P[i, j] = np.exp(-pairwise_distances[i, j] / sigma)

        P = (P + P.T) / (2 * X.shape[0])

        return P

    def compute_pairwise_distances(self, X):
        pairwise_distances = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i != j:
                    pairwise_distances[i, j] = np.linalg.norm(X[i] - X[j])

        return pairwise_distances

    def compute_low_dimensional_affinities(self, Y):
        pairwise_distances = self.compute_pairwise_distances(Y)
        Q = np.zeros((Y.shape[0], Y.shape[0]))
        denominator = np.sum(1 / (1 + pairwise_distances ** 2))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[0]):
                if i != j:
                    Q[i, j] = (1 + pairwise_distances[i, j] ** 2) ** -1 / denominator
        return Q

    def compute_gradient(self, P, Q, Y):
        dY = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[0]):
                if i != j:
                    dY[i] += 4 * (P[i, j] - Q[i, j]) * (Y[i] - Y[j]) / (1 + np.linalg.norm(Y[i] - Y[j]) ** 2)

        return dY