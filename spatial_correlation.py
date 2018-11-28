"""
Author: Paulo Cesar V. da Silva

This module calculates the average probability that a node escapes from its original
neighborhood.

Usage
-----
Call the program with one argument: the path of the input (config) file.

Input file example
------------------
> num_agents = 10
> num_steps = 100

> grid_width = 100
> grid_height = 100

> step_size = 3
> pj = 0.00  # Probability of a random jump
> interaction_radius = 5

{"pos": lambda ag: ag.pos}

"""

from toolbox.models import *
from toolbox.file_tools import read_inputs, get_output_file_name
import matplotlib.pyplot as plt

TAU_MAX = 50  # Number of time steps ahead to consider.


def tuple_dist(tup1, tup2, grid):
    """Calculates the Euclidean (periodic) distance between two 2D tuples.
    tup1 = (x1, y1)
    tup2 = (x2, y2)

    d = \sqrt((x1-x2)^2 + (y1-y2)^2)

    grid : mesa.ContinuousSpace
        The grid in which the positions are measured.
    """
    return grid.get_distance(tup1, tup2)
    # return np.sqrt((tup2[0]-tup1[0])**2 + (tup2[1]-tup1[1])**2)


def spatial_correlation_function(data, num_agents, num_steps, delta, grid):
    """Calculates the probability (mean over all t and agents)
    that an agent is still inside its own original neighborhood.

    Parameters
    ----------
    data : pd.DataFrame
        Collected positions of all agents.
    """
    # Array that stores the indicators (inside/outside neighborhood)
    indicator = np.empty((num_steps, num_agents, TAU_MAX))

    for t in range(num_steps):
        for agent_id in range(num_agents):
            # Gets position at time t
            r0 = data[t][agent_id]
            for tau in range(TAU_MAX):
                # Gets position at time t+tau
                r1 = data[t+tau+1][agent_id]
                # Attributes: 1 if it is still inside the circle.
                #             0 if it is not.
                if tuple_dist(r1, r0, grid) < delta:
                    indicator[t][agent_id][tau] = 1
                else:
                    indicator[t][agent_id][tau] = 0

    # Averages over all agents and time steps
    return indicator.mean(axis=(0, 1))


def export_arrays(tau_array, corr_array, argn=2):
    """Writes to a given file. If not passed, writes to standard file."""
    output_string = ""

    # Writes the content to exportt
    for tau, corr in zip(tau_array, corr_array):
        output_string += "{:d}\t{}\n".format(tau, corr)

    # Writes the content to file
    open(get_output_file_name(std_name="correlation_output.txt"), "w")\
        .write(output_string)


def main():

    # Reads the input config file
    input_dict = read_inputs(1)

    # Reads useful variables
    num_agents = int(input_dict["num_agents"])
    num_steps = int(input_dict["num_steps"])
    delta = float(input_dict["interaction_radius"])

    # Initializes the model
    model = FlatRandomWalk(*flat_random_walk_get_parameters(input_dict),
                           agent_reporters={"pos": lambda ag: ag.pos})  # Reporter that collects the agents' positions

    # Simulates the model for num_steps steps and TAU_MAX further.
    print("Simulating model")
    for t in range(num_steps+TAU_MAX):
        model.step()

    # Calculates the correlations.
    print("Calculating correlations")
    tau_array = np.arange(TAU_MAX+1)  # From 0 to TAU_MAX
    corr_array = np.empty(TAU_MAX+1)
    corr_array[0] = 1.  # For tau=0, correlation is always 1.

    corr_array[1:] = spatial_correlation_function(
            model.datacollector.get_agent_vars_dataframe()["pos"],
            num_agents,
            num_steps,
            delta,
            model.grid
        )

    # Exports the table
    export_arrays(tau_array, corr_array)

    # Plots the function
    print(corr_array)
    plt.scatter(tau_array, corr_array)
    plt.show()


if __name__ == "__main__":
    main()
