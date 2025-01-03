import numpy as np
from decimal import Decimal

class Recommender:
    def __init__(self, L, S, p):
        """_summary_

        Args:
        L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                         will give a like to a clip from genre i.
        S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i."""

        self.n_users = L.shape[1]
        self.n_genres = L.shape[0]

        self.L = L.copy()
        self.S = S.copy()
        self.p = p.copy()

        self.history = []
        self.p_history = [Decimal(p_j) for p_j in p]

        self.last_recommend = None

        self.pure_action = np.zeros_like(self.L)

        self.t = 0

    def recommend(self):
        """_summary_

        Returns:
        integer: The index of the clip that the recommender recommends to the user."""
        curr_p = np.array([(Decimal(self.p[j]) * self.p_history[j]) / sum(self.p_history) for j in range(len(self.p))],
                          dtype=np.float64)

        current_matrix = np.zeros_like(self.L)
        for _ in reversed(range(self.t, 15)):
            current_matrix = self.L + current_matrix * (self.L + (1 - self.L) * self.S)

        # Normalize the probabilities to ensure they sum to 1
        probabilities = np.dot(current_matrix, curr_p)
        probabilities /= np.sum(probabilities)

        # Use np.random.choice to choose an index based on the distribution
        recommendation = np.random.choice(self.n_genres, p=probabilities)

        self.last_recommend = recommendation
        return recommendation

    def update(self, signal):
        """_summary_

        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not.
                          It is 1 if the user liked the clip, and 0 otherwise."""
        self.t += 1