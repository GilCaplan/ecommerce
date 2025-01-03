import numpy as np


class Recommender:
    def __init__(self, L, S, p, mu, b_u, b_i, P, Q):
        self.L = L
        self.S = S
        self.p = p
        self.mu = mu
        self.b_u = b_u
        self.b_i = b_i
        self.P = P
        self.Q = Q
        self.belief = p.copy()  # Initialize belief with prior probabilities
        self.prev_clip = None

    @staticmethod
    def train_latent_factor_model(L, S, p, num_factors=10, learning_rate=0.01, reg=0.1, num_iterations=100):
        num_genres, num_users = L.shape
        mu = np.mean(L)
        b_u = np.zeros(num_users)
        b_i = np.zeros(num_genres)
        P = np.random.normal(scale=1. / num_factors, size=(num_users, num_factors))
        Q = np.random.normal(scale=1. / num_factors, size=(num_genres, num_factors))

        for _ in range(num_iterations):
            for u in range(num_users):
                for i in range(num_genres):
                    r_ui = L[i, u]
                    if not np.isnan(r_ui):
                        pred = mu + b_u[u] + b_i[i] + np.dot(P[u, :], Q[i, :])
                        error = r_ui - pred

                        b_u[u] += learning_rate * (error - reg * b_u[u])
                        b_i[i] += learning_rate * (error - reg * b_i[i])

                        P[u, :] += learning_rate * (error * Q[i, :] - reg * P[u, :])

        return mu, b_u, b_i, P

    def recommend(self):
        expected_likes = np.dot(self.L, self.belief)
        expected_stay = np.dot(self.S * (1 - self.L), self.belief)
        weighted_scores = expected_likes + expected_stay
        self.prev_clip = np.argmax(weighted_scores)
        return self.prev_clip

    def update(self, signal):
        if signal:
            likelihood = self.L[self.prev_clip]
        else:
            likelihood = (1 - self.L[self.prev_clip]) * self.S[self.prev_clip]

        self.belief = self.belief * likelihood
        self.belief /= np.sum(self.belief)
