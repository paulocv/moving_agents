"""
Call this program with one argument: the '.out' file of the simulation.
"""


from toolbox.file_tools import skip_file_header, read_inputs
import numpy as np
import matplotlib.pyplot as plt
import sys


NORMALIZE = True


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




def main():

    # Reads input parameters and simulation data.
    input_dict = read_inputs()
    sim_array = get_array_skip_header(sys.argv[1], dtype=int)

    # Mean over the different simulations
    mean_sim = np.mean(sim_array, axis=1)

    # Divides by the number of agents to get prevalence
    if NORMALIZE:
        mean_sim /= int(input_dict["num_agents"])

    plt.plot(mean_sim)
    plt.show()


if __name__ == "__main__":
    main()
