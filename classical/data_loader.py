import numpy as np
import string

# the function of loading data from the original file
# input data_file refers to the absolute path of the target file
# usage data = load_data('data.txt')
def load_data(data_file):
    f = open(data_file, 'r')
    contents = f.readlines()
    for line in range(len(contents)):
        contents[line] = contents[line].split('\t')
        for dim in range(len(contents[line])):
            contents[line][dim] = float(contents[line][dim])
    contents = np.array(contents)
    f.close()
    return contents



