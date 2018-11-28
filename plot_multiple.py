"""Makes multiple plots of datasets that are separated by a delimiter line.

Call with one argument: the input file.
"""

from toolbox.plot_tools import *
import sys

# ---------------------------
# INPUT PARAMETERS

usecols = (-3, -2)  # Column indexes that represent x and y data.
separator_line = "-----\n"
skiprows = 0


# labels = ["$\\delta = 0.01$",
#           "$\\delta = 0.1$",
#           "$\\delta = 0.5$",
#           "$\\delta = 1.0$",
#           "$\\delta = 2.0$",
#           "$\\delta = 10.0$"
#           ]
# labels = ["$p_j = 0.001$",
#           "$p_j = 0.1$",
#           "$p_j = 1.0$"
#           ]
# labels = ["$\\rho = {:0.2f}$".format(N/1600.)
#           for N in range(1600, 3800, 200)]
# labels = ["$\\rho = 0.50$",
#           "$\\rho = 0.75$",
#           "$\\rho = 1.00$",
#           "$\\rho = 1.25$",
#           "$\\rho = 1.50$",
#           "$\\rho = 1.75$",
#           "$\\rho = 2.00$"
#           ]
labels = ["$\\sigma = 0.00$",
          "$\\sigma = 0.01$",
          "$\\sigma = 0.05$",
          "$\\sigma = 0.1$",
          "$\\sigma = 0.5$"
          ]


x_label = "$\\beta$"
# x_label = "$p_j$"
# x_label = "$\\delta$"

y_label = "$i$"

x_lim = (0.000, 1.)  # Set to None for auto format
y_lim = (0.000, 1.)

xscale = "linear"
yscale = "linear"

# ----------------------------


def split_list(l, separator):
    """Splits a list at every occurrence of an element.
    Similar to str.split() method.
    Returns a list of lists.
    """
    result_list = [[]]

    for a in l:
        # Checks if the current element is a separator
        if a == separator:
            result_list += [[]]
        else:
            result_list[-1] += [a]

    return result_list


# -----------------------------
# Reads all the input file lines and splits into its parts (separated by 'separator_line')
plot_sets = split_list(open(sys.argv[1], "r").readlines()[skiprows:], separator_line)

# Here are given the plot commands
for data_set in plot_sets:
    line = plot_simple_from_str_list(data_set, usecols=usecols, fmt="o-")

# Plot adjustments
if labels is not None:
    plt.legend(labels)

plt.xlabel(x_label)
plt.ylabel(y_label)

if xscale is not None:
    plt.xscale(xscale)
if yscale is not None:
    plt.yscale(yscale)

if x_lim is not None:
    plt.xlim(x_lim)
if y_lim is not None:
    plt.ylim(y_lim)

plt.show()
