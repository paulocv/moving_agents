"""
Random generation tools that are specific to this project
"""
import numpy as np


DISTRIBUTIONS = {
    "delta": ["mean"],
    "poisson": ["mean", "std"],
}


def random_function_generator(distrib, **kwargs):
    """ Uses numpy random functions to create a single argument function
    that returns random float numbers according to a given distribution.
    """

    # First checks if the given type is at the dictionary of distributions.
    if distrib not in DISTRIBUTIONS:
        raise ValueError("Hey, probability distribution '{}' was not found "
                         "at the dictionary.\n"
                         "Possible types:\n"
                         "{}".format(distrib, list(DISTRIBUTIONS.keys())))

    # Single-value distribution
    if distrib == "delta":
        # Needs only the 'mean' value, which is actualy the only
        # possible value.
        mean = float(kwargs["mean"])

        def func(size=None):
            # If size is not passed, returns an array of one element.
            if size is None:
                return np.array([mean])
            else:
                return np.repeat(mean, size)

    # Adjustable Poisson distribution.
    elif distrib == "poisson":
        # Gather mean and standard deviation
        mean = float(kwargs["mean"])
        std = float(kwargs["std"])

        # Calculates the parameter of the distribution ($\lambda$) and
        # the "transformation coefficient", $r = \alpha*k$
        lamb = mean**2 /  std**2
        alpha = std**2 / mean

        def func(size=None):
            return np.array(alpha * np.random.poisson(lamb, size))

    #
    else:
        def func(size=None):
            return None
        raise ValueError("Hey, distribution '{}' could not be interpreted"
                         "".format(distrib))

    return func


def random_function_generator_from_dict(input_dict, prefix=""):
    """ Uses the 'random_function_generator' to create a random function, but
    gathers the parameters from a dictionary.
    """
    # Gathers the distribution type
    distrib = input_dict[prefix + "dist"]

    # Reads the necessary parameters, based on the required names contained in
    # DISTRIB dict
    arg_dict = {}
    for name in DISTRIBUTIONS[distrib]:
        arg_dict[name] = input_dict[prefix + name]

    return random_function_generator(distrib, **arg_dict)

