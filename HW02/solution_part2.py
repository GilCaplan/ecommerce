import numpy as np
from decimal import Decimal
import scipy


class Recommender:
    # Your recommender system class

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

        self.curr_p = p.copy()

        self.history = []
        self.p_history_and_type = [Decimal(p_j) for p_j in p]

        self.last_recommend = None

        self.all_matrices = np.zeros(L.shape + (15,))
        # try to find dominating option
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
        else:
            # if self.t < 4:
            # p_genre = scipy.special.softmax(np.dot(self.all_matrices[:, :, self.t], self.curr_p))
            # recommendation = np.random.choice(range(self.n_genres), p=p_genre)

            # p_likes = np.dot(self.L, self.curr_p)
            # random_variables = np.array([np.random.binomial(1, p_like_i) for p_like_i in p_likes])
            # recommendation = np.argmax(p_likes)

            # recommendation = np.argmax(np.dot(self.all_matrices[:, :, self.t], self.curr_p))
            # else:

            current_matrix = self.L.copy()
            for t in range(14, self.t, -1):

                means = np.dot(current_matrix, self.curr_p)
                means[means < np.median(means)] = -10000

                best_genre_likes = np.average(np.dot(current_matrix, self.curr_p),
                                              weights=scipy.special.softmax(means))

                # best_genre_likes = np.max(np.dot(current_matrix, self.curr_p))
                current_matrix = self.L + best_genre_likes * (self.L + (1 - self.L) * self.S)
            #
            recommendation = np.argmax(np.dot(current_matrix, self.curr_p))
            #
            self.last_recommend = recommendation
            return recommendation

    def update(self, signal):

        """_summary_

        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not.
                          It is 1 if the user liked the clip, and 0 otherwise."""

        self.history.append(signal)

        self.t += 1

        if signal:
            self.curr_p = self.curr_p * self.L[self.last_recommend, :]
        else:
            self.curr_p = self.curr_p * (1 - self.L[self.last_recommend, :]) * self.S[self.last_recommend, :]

        self.curr_p = self.curr_p / np.sum(self.curr_p)


# an example of a recommender that always recommends the item with the highest probability of being liked
class GreedyRecommender:
    def __init__(self, L, S, p):
        self.L = L
        self.S = S
        self.p = p

    def recommend(self):
        return np.argmax(np.dot(self.L, self.p))

    def update(self, signal):
        pass
