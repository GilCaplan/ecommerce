import numpy as np
from decimal import Decimal


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

        self.history = []
        self.p_history = [Decimal(p_j) for p_j in p]

        self.last_recommend = None

        self.best_recommendation = np.zeros((self.n_genres, self.n_users, 15))

        self.best_recommendation[:, :, 14] = self.L

        for t in reversed(range(14)):
            next_step = np.max(np.dot(self.best_recommendation[:, :, t + 1], self.p))
            self.best_recommendation[:, :, t] = next_step * ((1 - L) * S + 1) + L

        self.t = 0

    def recommend(self):
        """_summary_
        
        Returns:
        integer: The index of the clip that the recommender recommends to the user."""

        curr_p = np.array([(Decimal(self.p[j]) * self.p_history[j])/sum(self.p_history) for j in range(len(self.p))],
                          dtype=np.float64)

        # for t in reversed(range(self.t, 14)):
        #     next_step = np.max(np.dot(self.best_recommendation[:, :, t + 1], curr_p))
        #     self.best_recommendation[:, :, t] = next_step * ((1 - self.L) * self.S + 1) + self.L
        #
        # recommendation = np.argmax(np.dot(self.best_recommendation[:, :, self.t], curr_p))
        recommendation = np.argmax(np.dot(self.L, curr_p))

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
                self.p_history[j] *= Decimal(self.L[self.last_recommend, j])
            else:
                self.p_history[j] *= Decimal((1 - self.L[self.last_recommend, j]) * self.S[self.last_recommend, j])
    
    
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


