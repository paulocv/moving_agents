"""
WRITE FULL DOCUMENTATION

Simulation module.

Usage
-----
The program must be called with one argument:
    1. [Mandatory] Path for the input file.


Input file
----------


Output
------

"""

import sys
from toolbox.file_tools import read_inputs, skip_file_header
import numpy as np


STD_TRANSIENT_FRACTION = 0.75


def get_array_skip_header(filename, header_end="-----\n",
                          delimiter="\t", dtype=float):
    """Opens a file, skips its header (by ignoring until the header end
    string) and reads the data array after the header as a numpy array.

    Parameters
    ----------
    filename : str
        Name of the input file
    header_end : str
        Character that marks the end of the header. Must contain the '\n' at
        the end.
    delimiter : str
        Horizontal separator of each point.
    dtype : type
        Data type.

    Returns
    -------
    a : np.Array
        Array imported from file.
    """

    fp = open(filename, "r")
    skip_file_header(fp, header_end)
    a = np.loadtxt(fp, delimiter=delimiter, dtype=dtype)  # Reads the actual data.
    fp.close()

    return a


def get_stationary_sample_size(input_dict):
    """Gets the initial fraction of the sample that should be ignored
    to calculate the stationary prevalence. Returns the number of points
    that should actually be used in the analysis.
    This function is subject to future changes, as the fraction can be
    defined in a more dynamical fashion.
    """
    # Looks for the transient fraction on the input dictionary.
    # If it is not found, uses the standard value, stored in a
    # global variable.
    try:
        transient_frac = float(input_dict["transient_fraction"])
    except KeyError:
        transient_frac = STD_TRANSIENT_FRACTION

    # Gets the sample size (t_max or num_steps):
    num_steps = int(input_dict["num_steps"])

    # Returns the NUMBER (not fraction) of points
    return int(num_steps*(1.-transient_frac))


def calculate_prevalence(sim_array, num_t):
    """Averages over the simulations and over the iterations of the
    stationary stage.

    Parameters
    ----------
    sim_array : np.Array
        2D array with the simulation data.
        Lines (1st index) represent t (from 0 to num_steps),
        whereas collumns (2nd index) represent each simulation execution
        (from 0 to num_sim-1)
    num_t : int
        Number of points of the stationary state to be considered.

    """
    # First averages over the simulation executions.
    mean_stat = np.mean(sim_array[-num_t:], axis=1)

    # Then averages over the considered iterations.
    prevalence = np.mean(mean_stat)
    std_dev = np.std(mean_stat)

    return prevalence, std_dev


def main():

    # Reads the parameters of the simulation and the simulation data array.
    input_dict = read_inputs()
    sim_array = get_array_skip_header(sys.argv[1], dtype=int)

    # Gets the number of points of the stationary condition.
    # I.e., the number of iterations at the end of the sample that
    # will actually be considered to the analysis.
    num_t = get_stationary_sample_size(input_dict)

    # Calculates the mean and the standard deviation.
    mean, std = calculate_prevalence(sim_array, num_t)

    # Normalizes the results by the number of agents
    n_agents = int(input_dict["num_agents"])
    mean /= n_agents
    std /= n_agents

    # Writes the result to stdout.
    sys.stdout.write("{:0.6f}\t{:0.6f}\n".format(mean, std))


if __name__ == "__main__":
    main()
