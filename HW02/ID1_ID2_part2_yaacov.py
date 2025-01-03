import numpy as np
import scipy.stats

class Recommender:
    def __init__(self, L, S, p, beta_weight=0.5):
        """_summary_

        Args:
        L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                         will give a like to a clip from genre i.
        S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i.
        beta_weight (float): The weight given to the Beta distribution in the final recommendation.
                             (0 <= beta_weight <= 1)
        """

        self.n_users = L.shape[1]
        self.n_genres = L.shape[0]

        self.L = L.copy()
        self.S = S.copy()
        self.p = p.copy()

        self.curr_p = p.copy()

        self.history = []
        self.p_history_and_type = [p_j for p_j in p]

        self.last_recommend = None

        self.alpha = np.ones(L.shape)  # Initialize alpha parameters
        self.beta = np.ones(L.shape)   # Initialize beta parameters

        self.beta_weight = beta_weight

        self.all_matrices = np.zeros(L.shape + (15,))
        # Try to find dominating option
        self.all_matrices[:, :, 14] = self.L

        for t in range(13, -1, -1):
            self.all_matrices[:, :, t] = self.L + self.all_matrices[:, :, t + 1] * (self.L + (1 - self.L) * self.S)

        best_genres = np.argmax(self.all_matrices[:, :, 0], axis=1)

        self.dominating_action = None
        if np.all(best_genres == best_genres[0]):
            self.dominating_action = best_genres[0]

        self.best_action_in_advance = np.argmax(np.dot(self.all_matrices[:, :, 0], self.p))
        self.best_action_in_advance_likes = np.max(np.dot(self.all_matrices[:, :, 0], self.p))

        self.t = 0

    def recommend(self):
        """_summary_

        Returns:
        integer: The index of the clip that the recommender recommends to the user."""

        if self.dominating_action is not None:
            self.last_recommend = self.dominating_action
            return self.dominating_action

        sampled_L = np.zeros(self.L.shape)
        for i in range(self.n_genres):
            for j in range(self.n_users):
                sampled_L[i, j] = np.random.beta(self.alpha[i, j], self.beta[i, j])

        # Combine the matrices using the specified beta weight
        combined_L = self.beta_weight * sampled_L + (1 - self.beta_weight) * self.all_matrices[:, :, 0]
        combined_L = (combined_L - combined_L.min()) / (combined_L.max() - combined_L.min())

        if self.t < 4:
            recommendation = np.argmax(np.dot(combined_L, self.curr_p))
        else:
            current_matrix = combined_L.copy()
            for t in range(14, self.t, -1):
                means = np.dot(current_matrix, self.curr_p)
                means[means < np.median(means)] = -10000

                best_genre_likes = np.average(np.dot(current_matrix, self.curr_p), weights=scipy.special.softmax(means))

                current_matrix = combined_L + best_genre_likes * (combined_L + (1 - combined_L) * self.S)

            recommendation = np.argmax(np.dot(current_matrix, self.curr_p))

        self.last_recommend = recommendation
        return recommendation

    def update(self, signal):
        """_summary_

        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not.
                          It is 1 if the user liked the clip, and 0 otherwise."""

        self.history.append(signal)

        self.t += 1

        for j in range(self.n_users):
            if signal:
                self.alpha[self.last_recommend, j] += 1
                self.p_history_and_type[j] *= self.L[self.last_recommend, j]
            else:
                self.beta[self.last_recommend, j] += 1
                self.p_history_and_type[j] *= (1 - self.L[self.last_recommend, j]) * self.S[self.last_recommend, j]

        self.curr_p = np.array(
            [self.p_history_and_type[j] / sum(self.p_history_and_type) for j in range(len(self.p))],
            dtype=np.float64)
