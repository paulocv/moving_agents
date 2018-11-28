# File tools.

import sys
import os

SEP = "/"  # Use '/' for Linux and '\\' for Windows.


def remove_border_spaces(string):
    """ Strips the whitespace borders of a string.
    Used inside 'read_config_from_string' function.
    """
    if type(string) != str:
        raise TypeError(
            "Hey, a non-string object was passed to function "
            "'Remove_border_spaces'!")

    if string == '':
        return string

    while string[0] == ' ' or string[0] == '\t':
        string = string[1:]
        if string == '':
            return string

    while string[-1] == ' ' or string[-1] == '\t':
        string = string[:-1]
        if string == '':
            return string

    return string


def read_config_from_string(string, entry_char='>', attribution_char='=',
                            comment_char='#'):
    """Reads data stored as marked inputs from a string.
     It opens the string and looks for lines with 'entry_char'. Example:
        > option_name: value  #This is a comment

    The ':' can be replaced by any combination of characters specified
    as 'attribution_char' keyword argument.

    Inputs
    ----------
    string : str
        String from which the inputs are read.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.

    Returns
    ----------
    result_dictio : dict
        Dictionary with all the options read from file."""

    entry_char_len = len(entry_char)
    attr_char_len = len(attribution_char)

    # Gets each line of the string and stores into a list.
    string_content = string.splitlines()

    # Main loop along the lines of the string
    result_dictio = {}
    for line in string_content:

        # Gets only lines which have the entry character at the start
        if line[0:entry_char_len] != entry_char:
            continue

        # Line text processing
        # Ignores everything after a comment character
        line = line.split(comment_char)[0]
        # Eliminates the initial (entry) character
        line = line[entry_char_len:]

        # Separation between key and value
        # Finds where is the attribution char, which separates key from
        # value.
        attr_index = line.find(attribution_char)
        # If no attribution char is found, raises an exception.
        if attr_index == -1:
            raise ValueError(
                "Heyy, the attribution character '" + attribution_char +
                "' was not found in line: '" + line + "'")
        key = remove_border_spaces(line[:attr_index])
        value = remove_border_spaces(line[attr_index + attr_char_len:])

        # Finally adds the entry to the dictionary
        result_dictio[key] = value

    return result_dictio


def read_config_file(file_path, entry_char='>', attribution_char='=',
                     comment_char='#'):
    """Function that reads marked input files.
     It opens the file and looks for lines with 'entry_char'. Example:
        > option_name: value  #This is a comment

    The ':' can be replaced by any combination of characters specified
    as 'attribution_char' keyword argument.

    Inputs
    ----------
    file_path : str
        Name of the file to be read.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.

    Returns
    ----------
    result_dictio : dict
        Dictionary with all the options read from file.
    """
    # File opening and storing
    fp = open(file_path, 'r')
    file_str = fp.read()
    fp.close()

    return read_config_from_string(file_str, entry_char=entry_char,
                                   attribution_char=attribution_char,
                                   comment_char=comment_char)


def entry_string(key, value, entry_char=">", attribution_char="=",
                 end_char="\n"):
    """Converts a keyword and a value to an accepted input string for
    'read_config_file.

    Example:
    If key = "beta" and value = 0.5, the function returns the string:
    "> beta = 0.5"

    Inputs
    ----------
    key : str
        Keyword (name of the option/parameter). If not string, a
        conversion is attempted.
    value : str
        Value of the option/parameter. If not a string, a conversion
        is attempted.
    entry_char : str
        Character to start the line.
    attribution_char : str
        Character that separates the key from the value.
    end_char : str
        Character inserted at the end of the string.

    Returns
    ----------
    result_str : str
        String with an input line containing '> key = value'.
    """
    result_str = entry_char + " "
    result_str += str(key)
    result_str += " " + attribution_char + " "
    result_str += str(value)
    result_str += end_char
    return result_str


def write_config_string(input_dict, entry_char='>', attribution_char='=',
                        usekeys=None):
    """ Exports a dictionary of inputs to a string.

    Inputs
    ----------
    input_dict : dict
        Dictionary with the inputs to be exported to a string.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    usekeys : list
        Use that input to select the input_dict entries that you want
        to export.
        Inform a list of the desired keys. Default is the whole dict.

    Returns
    -------
    result_str : str
        String with the formated inputs that were read from the input_dict.
    """
    # Selects the desired entries of the input_dict
    if usekeys is not None:
        input_dict = {key: input_dict[key] for key in usekeys}

    result_str = ""

    for key, value in input_dict.items():
        result_str += entry_string(key, value, entry_char, attribution_char)

    return result_str


def write_config_file(input_dict, file_name, entry_char='>',
                      attribution_char='=', usekeys=None):
    """ Exports a dictionary of inputs to a file.

    Inputs
    ----------
    input_dict : dict
        Dictionary with the inputs to be exported to a string.
    file_name : str
        Path for the output file. Existing data on the file is
        erased!
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    """
    fp = open(file_name, "w")
    fp.write(write_config_string(input_dict, entry_char, attribution_char,
                                 usekeys))
    fp.close()


def skip_file_header(file_pointer, header_end="-----\n"):
    """Reads a file until it finds a 'header finalization' line.
    By standard, such line is '-----\n' (five -)

    Parameters
    ----------
    file_pointer : file
        Opened file.
    header_end : str
        String that marks the end of a header section.
        Must contain the '\n' at the end.
    """
    # Reads file line once
    line = file_pointer.readline()

    while line:  # Reads until eof
        # Checks if the line is a header ender
        if line == header_end:
            return
        line = file_pointer.readline()

    # If EOF is reached without finding the header end, an error is raised.
    raise EOFError("Hey, I did not find the header ending string on file:\n"
                   "File: '{}'\n"
                   "Ending str:'{}'\n".format(file_pointer.name, header_end))


def read_inputs(argn=1):
    """Reads the input file, passed as the first argument from
    program call. Also checks if the argument was passed and if the
    file path exists.

    Parameters
    ----------
    argn : int
        Position of the input file name in sys.argv. Standard is 1.

    Returns
    ----------
    input_dict : dict
        Dictionary with all inputs read from file.
    """
    if len(sys.argv) < argn+1:
        raise IOError("Hey, no input file was passed as argument to"
                      " the program!!")
    if not os.path.exists(sys.argv[argn]):
        raise FileNotFoundError("Input file '{}' not found.".
                                format(sys.argv[argn]))
    return read_config_file(sys.argv[argn], attribution_char='=')


def get_output_file_name(argn=2, std_name='output.txt'):
    """ Tries to read the output file from argv[argn].
    If argv[argn] does not exist, the output file is set to sdt_name,
    whose standard value is 'output.txt'.

    Parameters
    ----------
    argn : int
        Index of the output file name in sys.argv. Standard is 2.
    std_name : str
        Name of the output file, in case that it is not passed in argv.

    Returns
    ----------
    name : str
        Name of the output file.
    """
    try:
        name = sys.argv[argn]
    except IndexError:
        name = std_name
        print("Warning: no output file name received. Output will be"
              " written to '%s'." % name)
    return name


def get_output_folder_name(argi=2, root_folder=""):
    """Reads a folder path from argv (argv[2] by standard).
    Adds the separator character (/, \\) if it was forgotten.
    Checks if the folder exists, and creates it otherwise.

    If the corresponding position in argv is not informed, asks for the
    user the path of the folder, starting from a given root folder.

    """
    # First tries to read the output folder name from argv[2]
    try:
        output_folder = sys.argv[argi]
    except IndexError:
        # If argv[argi] was not passed, asks the user for the output folder.
        output_folder = root_folder
        output_folder += input("Output folder path was not informed. Please inform:\n"
                               "{}".format(root_folder))

    # Adds the SEP (/ or \\) character to the end of the folder name.
    if output_folder[-len(SEP):] != SEP:
        output_folder += SEP

    # Checks if the folder does not exist. Creates it, in this case.
    if not os.path.exists(output_folder):
        os.system("mkdir -p '{}'".format(output_folder))

    return output_folder


def read_csv_names(string):
    """Reads multiple strings separated by commas and removes border spaces.
    Example:
        "beta, pj ,  num_steps" --> ['beta', 'pj', 'num_steps']
    """
    return [remove_border_spaces(name) for name in string.split(',')]
