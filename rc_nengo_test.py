# based off https://github.com/tcstewar/testing_notebooks/blob/master/Reservoir.ipynb

import nengo
import scipy.io

def main(t_len, dims, n_classes, val_func, prob_type):
    