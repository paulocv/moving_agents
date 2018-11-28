
# from toolbox.rand_tools import *
# import matplotlib.pyplot as plt
from toolbox.file_tools import *
from toolbox.flat_models import *

input_dict = read_inputs()

inputs = mtrumor_flat_radvar_random_walk_get_parameters(input_dict)
model = MTrumorFlatRadvarRandomWalk(*inputs)

for ag in model.agents():
    print(ag.delta)

