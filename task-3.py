from typing import Callable

import numpy as np
from math import log, exp
from scipy.stats import rv_continuous, uniform
import matplotlib.pyplot as plt


def pdf(x):
    if x <= 0:
        return 0
    return 3 * x**2 * np.exp(-x ** 3)


class CustomDistribution(rv_continuous):
    def _pdf(self, x, **kwargs):
        return pdf(x)


def plot_distribution(rand: Callable[[], float], n_tries: int, color: str):
    plt.xlabel(f'Number of tries = {n_tries}')
    plt.hist([rand() for _ in range(n_tries)], color=color, bins=200)


def plot_density():

    def p(x):
        return 3 * x**2 * np.exp(-x ** 3)

    x = np.arange(0., 10., 0.1)
    plt.xlabel('y = p(x)')
    plt.plot(x, p(x))


def inherit_generator():
    distribution = CustomDistribution()

    def gen():
        x = distribution.rvs()
        if x >= 3:
            # some anomaly happens and too large values are generated sometimes
            x = distribution.rvs()
        return x

    return gen


def inverse_generator():
    distribution = uniform()

    def gen():
        return -np.cbrt(log(1 - distribution.rvs()))

    return gen


def rejection_generator():
    distribution = uniform()

    def gen():
        while True:
            x = distribution.rvs() * 3
            y = distribution.rvs() * 1.5
            if y <= pdf(x):
                return x

    return gen

if __name__ == '__main__':
    #plot_density()
    plot_distribution(inherit_generator(), 5000, 'blue')
    #plot_distribution(inverse_generator(), 5000, 'red')
    #plot_distribution(rejection_generator(), 5000, 'green')
    plt.show()
