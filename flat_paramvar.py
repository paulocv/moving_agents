"""
WRITE FULL DOCUMENTATION

Simulation of multiple parameters module.

Usage
-----
The program must be called with one or two arguments:
    1. [Mandatory] Path for the input file.
    2. [Optional] Path for the output folder.


Input file
----------


Output
------

"""


from toolbox.file_tools import read_inputs, write_config_string, \
     write_config_file, get_output_folder_name, read_csv_names
import os
import sys
import datetime
import itertools
# import math
import time

# Dictionary of possible models and executions, used for the program calling.
MODEL_DICT = {
    "flat_sis": {
        "simulator": "flat_sis_simulate.py",
        "analyzer": "flat_sis_analyze.py"
    },
    "flat_sir": {
        "simulator": "flat_sir_simulate.py",
        "analyzer": "flat_sir_analyze.py"
    },
    "flat_mtrumor": {
        "simulator": "flat_mtrumor_simulate.py",
        "analyzer": "flat_sir_analyze.py"
    },
    "flat_mtrumor_radvar": {
        "simulator": "flat_mtrumor_radvar_simulate.py",
        "analyzer": "flat_sir_analyze.py"
    }
}


# Other important global parameters
STD_FILE_PREFIX = "sim"
SEP = "/"  # Folder separator. / = Unix, \\ = Windows.
RESULTS_ROOT_FOLDER = "results/"
TEMP_FILE_NAME = "temp_single_input.tmp"


def get_file_prefix(input_dict):
    """Gets the prefix string for the output files (including summary file).
    If file_prefix is not found in the input dict, a standard string is used.
    """
    try:
        file_prefix = input_dict["file_prefix"]
    except KeyError:
        file_prefix = STD_FILE_PREFIX
        print("No file_prefix found on the input file. Using '{}' as prefix."
              "".format(file_prefix))
    return file_prefix


def str_to_list(string, key_name=""):
    """Evaluates a string as a list. Checks if border characters are
    '[' and ']', to avoid bad typing.

    key_name : string (optional)
        Name of the parameter. Useful for error message.
    """
    if string[0] == '[' and string[-1] == ']':
        return list(eval(string))
    else:
        raise ValueError("Hey, bad parameter or list of parameters"
                         " in {} = {}".format(key_name,
                                              string))


def get_variable_parameters(input_dict):
    """
    Reads the "vary_parameters" input (i.e., the list of names of the
    parameters that must vary during all the simulations.
    Also interpret each variable parameter as a list.

    Returns a dictionary with the varying parameter names as keys and
    the corresponding lists of values as values.
    """

    # Reads the names of the parameters that will vary
    var_param_names = read_csv_names(input_dict['vary_parameters'])

    # For each varying parameter, tries to interpret the corresponding list
    # of values in the input_dict.
    var_param_dict = dict()
    for name in var_param_names:
        # Interprets the input as a list.
        var_param_dict[name] = str_to_list(input_dict[name], name)

    return var_param_dict


def summary_file_overwriting_check(file_path):

    # Checks if the file already exists and prompt an action to the user.
    if os.path.exists(file_path):
        answer = input("\nWarning: summary file '{}' already exists, meaning that "
                       "you may be about to overwrite an older simulation.\n"
                       "Do you want to stop ('s'), rename ('r') or overwrite it"
                       "anyway ('o')?\n".format(file_path))
        # If the user wants to overwrite it anyway:
        if answer == "o":
            return file_path
        # If the user wants to rename the file
        elif answer == "r":
            file_path = os.path.dirname(file_path) + SEP
            file_path += input(file_path)
            return file_path
        # If the user wants to stop
        elif answer == "s":
            print("Quitting now...")
            quit()
        else:
            print("What did you type dude!?? I'll quit anyway, dumb...")
            quit()

    # If file does not exist, return its path anyway
    return file_path


def write_summary_file_header(out_file_name, input_dict, param_var_dict):

    # If summary file exists, the current simulation may be overwriting
    # an older one. User is prompted to what action to take.
    out_file_name = summary_file_overwriting_check(out_file_name)

    fp = open(out_file_name, "w")

    # Writes the program name, the input file and the current date/time.
    fp.write("Output from '{0} {1}'.\n".format(__file__, sys.argv[1]))
    fp.write("{}\n\n".format(datetime.datetime.now()))

    # Writes the data from the mult-input dictionary.
    fp.write(write_config_string(input_dict))

    # Writes the header of the data.
    fp.write('\n')
    for name in param_var_dict:
        fp.write(name + '\t')
    fp.write("prevalence\tstd_dev\n")

    # "End of header" line
    fp.write("-----\n")

    fp.close()


def build_single_input_dict(mult_input_dict, keys_list, values_list):
    """Returns a single-input dict from a mult-input dict, for the given
    set of variable parameters.

    Parameters
    ----------
    mult_input_dict : dict
        Original dictionary with mult-inputs (sets of variable parameters).
    keys_list : list
        List of names of the variable parameters.
    values_list : list
        List of values of the variable parameters. Must follow the same order
        of the keys_list.
    """
    single_input_dict = mult_input_dict.copy()

    # For each variable parameter, replaces the entry on the mult-input,
    # transforming it into a single input dict.
    for i in range(len(keys_list)):
        single_input_dict[keys_list[i]] = values_list[i]

    return single_input_dict


def build_output_file_name(parameter_values, output_folder, file_prefix, index):
    """Builds the .out file name of the current simulation."""
    out_file_name = output_folder + file_prefix + "_" + str(index)

    # Inserts each value.
    for value in parameter_values:
        out_file_name += "_{:0.4f}".format(value).replace(".", "p")

    out_file_name += ".out"
    return out_file_name


def run_simulations(mult_input_dict, var_param_dict, output_folder,
                    summary_file, file_prefix):
    """Main loop that runs the simulations for each set
    of parameters.

    Parameters
    ----------
    mult_input_dict : dict
        Mult-input dictionary, as read from the input file.
    var_param_dict : dict
        Dictionary of variable parameters.
        Keys are the names of the variables, whereas values are the lists
        of values that will be used on the simulation.
    output_folder : str
        Path for the output folder.
    summary_file : str
        Path for the summary output file. This file stores only the results of
        the analysis of each simulation.
    file_prefix : str
        Prefix to be used for every output file (including summary file).
    """
    # Reads, from the input dict, the model that should be used
    model_names = MODEL_DICT[mult_input_dict["model"]]

    # List of lists, each one with the values of one parameter.
    values_list = list(var_param_dict.values())
    keys_list = list(var_param_dict.keys())  # List of parameter names

    # Gets the number of simulations that will be executed:
    num_sim = len(list(itertools.product(*values_list)))

    # Initial screen feedback
    print("Paramvar for model: " + mult_input_dict["model"])
    print("Varying parameters: {}".format(keys_list))

    # Iterates over the cartesian product of the parameters.
    for i_sim, parameter_values in enumerate(itertools.product(*values_list)):

        # Creates a single-input dict with the current values of the variables.
        sing_input_dict = build_single_input_dict(mult_input_dict,
                                                  keys_list,
                                                  parameter_values)

        # Writes a temporary file with the single-inputs (for one simulation run)
        write_config_file(sing_input_dict, TEMP_FILE_NAME)

        # Builds the name of the current simulation output file
        out_file_name = build_output_file_name(parameter_values, output_folder,
                                               file_prefix, i_sim)

        # Runs the simulation and analyzes the results
        print("\n[{} of {}] Running for parameters: {}"
              "".format(i_sim, num_sim, parameter_values))

        if os.system("python " + model_names["simulator"] + " " +
                     TEMP_FILE_NAME + " " +  # Temporary input file
                     out_file_name  # Simulation output file
                     ):
            return

        # Writes the current parameters to the summary file
        fp = open(summary_file, "a")
        for value in parameter_values:
            fp.write(str(value) + "\t")
        fp.close()

        # Calls the analyzer module.
        if os.system("python " + model_names["analyzer"] + " " +
                     out_file_name + " " +  # Simulation file
                     ">> " + summary_file
                     ):
            return


def main():
    # Reads the mult-input dict.
    mult_input_dict = read_inputs(1)

    # Reads the parameters that will be varied.
    var_param_dict = get_variable_parameters(mult_input_dict)

    # Gets the file prefix. If not informed, standard is used.
    file_prefix = get_file_prefix(mult_input_dict)

    # Gets and creates the output folder.
    output_folder = get_output_folder_name(root_folder=RESULTS_ROOT_FOLDER)

    # Creates and initializes the simulation summary file.
    summary_file_path = output_folder + file_prefix + "_summary.txt"
    write_summary_file_header(summary_file_path, mult_input_dict, var_param_dict)

    # Main loop.
    t0 = time.time()
    run_simulations(mult_input_dict, var_param_dict, output_folder,
                    summary_file_path, file_prefix)
    print("\nTotal time: {:0.2f} s".format(time.time()-t0))


if __name__ == "__main__":
    main()
