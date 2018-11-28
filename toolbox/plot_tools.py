"""
WRITE FULL DOCUMENTATION

Plot tools module. Feasible to use with interactive python.


"""

import matplotlib.pyplot as plt
import numpy as np


def plot_simple_from_file(fname, fmt=None, skiprows=0, usecols=None, delimiter=None):
    """Plots data from file.

    Parameters
    ----------
    fname : str
        Path for the file.
    skiprows : int
        Header lines to skip.
    fmt : str
        Format string to be used in 'plot' function
    delimiter : str
        Data delimiter in file.
    usecols : int or sequence
        Collumns to be used on import. If not informed, it reads the two
        last columns and uses the first as x and the second as y.

    Returns
    -------
    line : plt.line2D
    """

    # Imports data
    data = np.loadtxt(fname, skiprows=skiprows, usecols=usecols,
                      delimiter=delimiter, unpack=True)
    # Gets the two last columns that are read.
    x, y = data[-2:]

    if fmt is None:
        return plt.plot(x, y)
    else:
        return plt.plot(x, y, fmt)


def plot_simple_from_str_list(strlist, fmt=None, skiprows=0, usecols=None, delimiter='\t',
                              dtype=float):
    """Plots data from a list of strings, each one representing a line.
    The string list is expected as read by method file.readlines().

    Parameters
    ----------
    strlist : list
        String list with the data lines.
    skiprows : int
        Header lines to skip.
    fmt : str
        Format string to be used in 'plot' function
    delimiter : str
        Data delimiter in file.
    usecols : int or sequence
        Collumns to be used on import. If not informed, it reads the two
        last columns and uses the first as x and the second as y.
    dtype : cast or tuple
        Type of the entries. May be passed as a single type cast or a tuple
        of type casts, one for x and the other for y.

    Returns
    -------
    line : plt.line2D
    """

    if usecols is None:
        usecols = (-2, -1)

    # If not none, usecols must have two elements: x and y columns.
    if len(usecols) != 2:
        raise ValueError("Hey, usecols argument must have two elements. Found"
                         "{}.".format(len(usecols)))

    # If datatype is a single value, converts into a tuple.
    try:
        dtype = dtype[:2]  # Gets the first two entries.
    except TypeError:
        dtype = (dtype, dtype)

    # Number of points.
    n = len(strlist) - skiprows
    x = np.empty(n)
    y = np.empty(n)

    for i, line in enumerate(strlist[skiprows:]):
        # Removes the "\n" and splits by the delimiter.
        line = line[:-1].split(delimiter)

        # Checks if the line actually contains the desired collumns
        if len(line) < min(abs(usecols[0]), abs(usecols[1])):
            raise IndexError("Hey, line '{}' from file does not contain"
                             "the required number of columns.".format(line))

        # Adds the points to the arrays
        x[i] = dtype[0](line[usecols[0]])
        y[i] = dtype[1](line[usecols[1]])

    # Returns the line2D object
    if fmt is None:
        return plt.plot(x, y)
    else:
        return plt.plot(x, y, fmt)


def plot_errorbar_from_str_list(strlist, axes, fmt=None, skiprows=0, usecols=None, delimiter='\t',
                              dtype=float):
    """Plots data from a list of strings, each one representing a line.
    The string list is expected as read by method file.readlines().

    The method expects to read three columns:
    * x_array
    * y_array
    * y_error

    Parameters
    ----------
    strlist : list
        String list with the data lines.
    skiprows : int
        Header lines to skip.
    fmt : str
        Format string to be used in 'plot' function
    delimiter : str
        Data delimiter in file.
    usecols : int or sequence
        Collumns to be used on import. If not informed, it reads the two
        last columns and uses the first as x and the second as y.
    dtype : cast or tuple
        Type of the entries. May be passed as a single type cast or a tuple
        of type casts, one for x and the other for y.

    Returns
    -------
    line : plt.line2D
    """

    if usecols is None:
        usecols = (-3, -2, -1)

    # If not none, usecols must have two elements: x and y columns.
    if len(usecols) != 3:
        raise ValueError("Hey, usecols argument must have three elements. Found"
                         "{}.".format(len(usecols)))

    # If datatype is a single value, converts into a tuple.
    try:
        dtype = dtype[:3]  # Gets the first three entries.
    except TypeError:
        dtype = (dtype, dtype, dtype)

    # Number of points.
    n = len(strlist) - skiprows
    x = np.empty(n)
    y = np.empty(n)
    y_err = np.empty(n)

    for i, line in enumerate(strlist[skiprows:]):
        # Removes the "\n" and splits by the delimiter.
        line = line[:-1].split(delimiter)

        # Checks if the line actually contains the desired collumns
        if len(line) < min(abs(l) for l in usecols):
            raise IndexError("Hey, line '{}' from file does not contain"
                             "the required number of columns.".format(line))

        # Adds the points to the arrays
        x[i] = dtype[0](line[usecols[0]])
        y[i] = dtype[1](line[usecols[1]])
        y_err[i] = dtype[2](line[usecols[2]])

    # Returns the line2D object
    if fmt is None:
        return axes.errorbar(x, y, yerr=y_err)
    else:
        return axes.errorbar(x, y, yerr=y_err, fmt=fmt)

