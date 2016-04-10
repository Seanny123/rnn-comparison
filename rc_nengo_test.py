# based off https://github.com/tcstewar/testing_notebooks/blob/master/Reservoir.ipynb

import nengo
import scipy.io

def main(t_len, dims, n_classes, val_func, prob_type):

    # make a model and run it to get spiking data
    train_model = nengo.Network()

    # pass the spiking data and the target to a solver to get decoding weigths
    # save the decoding weigths?

    # run the test data with new decoding weights
    # set the decoding weights as transforms on a function
