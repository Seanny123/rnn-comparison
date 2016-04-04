import lasagne
import scipy.io

def main(t_len, dims, n_classes, val_func, prob_type):
    """given dimensions, t_len, """

    # if not discrete, discretize input signal