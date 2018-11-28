"""
WRITE FULL DOCUMENTATION

Simulation module.

Usage
-----
The program must be called with one or two arguments:
    1. [Mandatory] Path for the input file.
    2. [Optional] Path for the output file.

If second argument is not passed, output is written to 'output_flat_sir.out'

Input file
----------


Output
------

"""

from toolbox.file_tools import read_inputs, get_output_file_name, \
    write_config_string
from toolbox.flat_models import MTrumorFlatRandomWalk, \
    mtrumor_flat_random_walk_get_parameters
import numpy as np
import sys
import datetime
from time import time


def simulate_n_times(model, input_dict):
    """DESCRIPTION!!!

    Parameters
    ----------
    model : MTrumorFlatRandomWalk
        Model
    input_dict : dict
        Input dictionary.
    """
    # Parameter unpacking

    num_simulations = int(input_dict["num_simulations"])
    infected_fraction = float(input_dict["initial_infected"])

    # Initialization of the 2d array that stores simulation results
    # First index is for simulation (from 0 to num_sim-1)
    # Second index is for the iteration t (from 0 to num_steps)
    i_2d_array = np.empty((num_simulations, 1), dtype=int)

    # Array to store the execution time of each simulation.
    exec_time = np.empty(num_simulations)

    # Simulation loop
    print("Simulation has begun.")
    for n in range(num_simulations):
        # Execution time: initial point
        t0 = time()

        # Reinitializes the model and requests one simulation
        model.reinitialize_epidemic_states(infected_fraction)
        model.reposition_agents_at_random()
        # i_2d_array[n][0] = model.simulate_until_dies()
        i_2d_array[n][0], t, trials = model.simulate_and_retry(infected_fraction,
                                                               max_t=800,
                                                               max_trials=3,
                                                               return_numsteps=True)

        # Real time feedback of the simulation progress
        dt = time() - t0
        print("{} of {} ({:0.3f} s) - ".format(n+1, num_simulations, dt), end="")
        print("{} steps, {} trial".format(t, trials))
        exec_time[n] = dt

    # Feedback: shows the average and total time during simulations
    print("Simulation time: {:0.2f} s".format(exec_time.sum()))
    print("Average time for each sim: {:0.2f} s".format(exec_time.mean()))

    return i_2d_array


def write_file_header(out_file_name, input_dict):

    fp = open(out_file_name, "w")

    # Writes the program name, the input file and the current date/time.
    fp.write("Output from '{0} {1}'.\n".format(__file__, sys.argv[1]))
    fp.write("{}\n\n".format(datetime.datetime.now()))

    # Writes the data from the input dictionary.
    fp.write(write_config_string(input_dict))

    # "End of header" line
    fp.write("-----\n")

    fp.close()


def write_results_to_file(out_file_name, i_2d_array, input_dict,
                          delimiter="\t"):

    # The i_array is transposed, so that lines correspond to t
    # and collums correspond to each simulation execution
    out_array = i_2d_array.transpose()

    fp = open(out_file_name, "ab")  # Has to be binary (stackoverflow)
    np.savetxt(fp, out_array, delimiter=delimiter, fmt="%d")
    fp.close()


def main():

    # Reads the input file
    input_dict = read_inputs(1)

    # Gets the output file name, if it was passed.
    out_file_name = get_output_file_name(std_name="output_flat_mtrumor.out")

    # Creates an SIR-model instance.
    model = MTrumorFlatRandomWalk(*mtrumor_flat_random_walk_get_parameters(input_dict))

    # Simulation (gets the number of infected in each simulation and each t.
    i_2d_array = simulate_n_times(model, input_dict)

    # Output process
    write_file_header(out_file_name, input_dict)
    write_results_to_file(out_file_name, i_2d_array, input_dict)

if __name__ == "__main__":
    main()
