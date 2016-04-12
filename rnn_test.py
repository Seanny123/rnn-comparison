import lasagne
import scipy.io

def main(t_len, dims, n_classes, val_func, prob_type):
    """given dimensions, t_len, """

    # if not discrete, discretize input signal?
    # train up using Lasagne

    # make number of batches equal to one sig or multiple sigs?
    N_BATCH=10

    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, t_len/dt, dims))
    # do I need a mask or not?

    l_rec = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

    l_out = lasagne.layers.DenseLayer(l_rec,  num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    target_values = T.vector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    # The network output will have shape (n_batch, 1); let's flatten to get a
    # 1-dimensional vector of predicted values
    predicted_values = network_output.flatten()
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)


    # run the test in Nengo