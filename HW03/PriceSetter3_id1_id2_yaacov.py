import numpy as np
from time import perf_counter

import scipy.special
from scipy.stats import beta as beta_dist

ALPHA_VALUES = [i for i in range(1, 11)]
BETA_VALUES = [i for i in range(1, 11)]


class PriceSetter3:
    def __init__(self, rounds):
        """
        Initialize the price setter.
        In this settings, the values of the costumers is distributed according to a beta distribution with unknown parameters.

        Args:
            rounds (int): the number of rounds to simulate
        """

        self.a = 1
        self.b = 1

        self.results = {}
        self.last_price = None
        self.last_outcome = None

        def beta_cdf(x, a, b):
            return scipy.special.betainc(a, b, x)

        self.beta_cdf = beta_cdf

        self.xdata_not_normalized = []
        self.ydata = []

        self.trials_result = {}

        self.continue_update = True

        self.x_values = np.arange(1, 11) / 10
        self.y_bins = np.zeros_like(self.x_values)

        self.log_likelihood = np.zeros((len(ALPHA_VALUES), len(BETA_VALUES)))
        self.best_values = np.zeros((len(ALPHA_VALUES), len(BETA_VALUES)))

        for i, alpha in enumerate(ALPHA_VALUES):
            for j, beta in enumerate(BETA_VALUES):
                self.best_values[i, j] = scipy.optimize.minimize_scalar(lambda p: -(1 - beta_dist.cdf(p, alpha, beta)) * p, bounds=(0, 1)).x

        self.N_explore = 300

    def set_price(self, t):
        """
        Return the price at time t.

        Args:
            t (int): the time period

        Returns:
            float: the price at time t
        """
        raveled_log_likelihood = np.ravel(self.log_likelihood)
        T = 0.5 + 1/(t + 1)
        probabilities = scipy.special.softmax(raveled_log_likelihood/T)

        raveled_ind = np.random.choice(range(len(ALPHA_VALUES) * len(BETA_VALUES)), p=probabilities)
        unraveled_ind = np.unravel_index(raveled_ind, self.log_likelihood.shape)

        alpha = ALPHA_VALUES[unraveled_ind[0]]
        beta = BETA_VALUES[unraveled_ind[1]]

        price = self.best_values[unraveled_ind]
        # raveled_ind = self.log_likelihood.argmax()
        # unraveled_ind = np.unravel_index(raveled_ind, self.log_likelihood.shape)
        #

        #
        # alpha_i = unraveled_ind[0]
        # beta_i = unraveled_ind[1]
        #
        # # if t < self.N_explore:
        # #     price = np.random.rand()
        # # else:
        # price = np.random.normal(self.best_values[alpha_i, beta_i], 0.25)
        # if price < 0:
        #     price = 0.01
        # if price > 1:
        #     price = 0.99
        #
        self.last_price = price
        return price

    def update(self, t, outcome):
        """
        Update the price setter based on the outcome of the previous period.

        Args:
            t (int): the time period
            outcome (int): the outcome of the previous period - true if the product was sold, false otherwise
        """
        # if t < self.N_explore:
        if True:
            self.last_outcome = outcome
            self.results[self.last_price] = outcome

            price = self.last_price

            outcome_indicator = int(outcome)
            for i, alpha in enumerate(ALPHA_VALUES):
                for j, beta in enumerate(BETA_VALUES):
                    F_price = scipy.special.betainc(alpha, beta, price)
                    diff = (1 - outcome_indicator) * np.log(F_price) + outcome_indicator * np.log(1 - F_price)

                    if not np.isnan(diff):
                        self.log_likelihood[i, j] += diff


def simulate(simulations, rounds):
    """
    Simulate the game for the given number of rounds.

    Args:
        rounds (int): the number of rounds to simulate the game

    Returns:
        float: the average revenue of the seller
    """

    simulations_results = []
    for _ in range(simulations):

        alpha = np.random.choice(ALPHA_VALUES)
        beta = np.random.choice(BETA_VALUES)
        start = perf_counter()
        price_setter = PriceSetter3(rounds)
        end = perf_counter()
        if end - start > 1:
            raise Exception("The initialization of the price setter is too slow.")
        revenue = 0

        for t in range(rounds):
            costumer_value = np.random.beta(alpha, beta)
            start = perf_counter()
            price = price_setter.set_price(t)
            end = perf_counter()
            if end - start > 0.3:
                raise Exception("The set_price method is too slow.")
            if costumer_value >= price:
                revenue += price

            start = perf_counter()
            price_setter.update(t, costumer_value >= price)
            end = perf_counter()
            if end - start > 0.3:
                raise Exception("The update method is too slow.")

        simulations_results.append(revenue)

    return np.mean(simulations_results)


if __name__ == "__main__":
    np.random.seed(0)
    print(simulate(1000, 1000))
