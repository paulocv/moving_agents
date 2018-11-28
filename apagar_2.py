
from numpy.random import poisson, normal
import matplotlib.pyplot as plt


MU = 10.
SIGMA = 3.
N = 50000


def func_generator(mu, sigma, type="normal"):
    lamb = mu**2/sigma**2
    alpha = sigma**2/mu


    if type=="normal":
        def func(size):
            return normal(mu, sigma, size=size)

    elif type=="poisson":
        def func(size=None):
            return alpha*poisson(lamb, size)

    else:
        func = 0

    return func

rand = func_generator(MU, SIGMA)
plt.hist(rand(N))

rand = func_generator(MU, SIGMA, "poisson")
plt.hist(rand(N))

plt.show()

