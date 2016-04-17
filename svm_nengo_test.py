# based off of Aaron's SVM

import nengo
from nengolib.synapses import HeteroSynapse, Bandpass
import scipy.io
from sklearn import svm

from constants import *
from dataman import *
from post import *

def multisynapse(src, dest, sub_features):

    synapses = [Bandpass(freq, Q) for (freq, Q) in sub_features]
    syn = nengo.Node(size_in=1, output=HeteroSynapse(synapses, dt=dt))

    nengo.Connection(src, syn, synapse=None)
    nengo.Connection(
        syn, dest.neurons, synapse=None,
        function=lambda x, transform=np.squeeze(dest.encoders): transform*x)

def main(t_len, dims, n_classes, dataset, testset):

    n_neurons = 200
    tau = 0.1
    # setup the feature craziness
    freq_range = (0, 500)
    Q_range = (2, 50)
    features_per_dim = 400
    feat_pops = []
    feat_list = []

    # make a model and run it to get spiking data
    # as acquired when passed through multiple bandpass filters
    train_model = nengo.Network()
    with train_model:
        feed_net = create_feed_net(dataset[0], dataset[1], t_len, dims, n_classes)

        state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED)
        state.add_neuron_output()

        nengo.Connection(feed_net.q_in, state.input, synapse=None)


        for dim in range(dims):
            # this declaration needed for the multisynapse transform
            encoders = nengo.dists.UniformHypersphere(
                        surface=True).sample(features_per_dim, 1)
            feat_pop = nengo.Ensemble(features_per_dim, 1, encoders=encoders,
                seed=SEED+dim)
            feat_pops.append(feat_pop)

            sub_features = zip(
                nengo.dists.Uniform(*freq_range).sample(features_per_dim),
                nengo.dists.Uniform(*Q_range).sample(features_per_dim)
            )
            feat_list.append(sub_features)
            multisynapse(state.ensembles[dim], feat_pop, sub_features)

        p_sig = nengo.Probe(feed_net.q_in, synapse=None)
        p_target = nengo.Probe(feed_net.get_ans, synapse=None)

        p_normal = nengo.Probe(state.output, synapse=tau)

        p_features = [
            nengo.Probe(
                feat_pop.neurons, sample_every=0.1, synapse=tau)
            for feat_pop in feat_pops]

    print("training simulation start")
    sim_train = nengo.Simulator(train_model)
    with sim_train:
        sim_train.run((t_len + PAUSE)*dataset[0].shape[0])
    print("training simulation done")

    plt.plot(sim_train.trange(), sim_train.data[p_sig], alpha=0.6)
    plt.plot(sim_train.trange(), sim_train.data[p_target], alpha=0.6)
    plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
    plt.ylim(-1.1, 1.1)
    #plt.legend()
    plt.show()

    # pass the feature data and the target to train an SVM
    print("Training SVM")
    # format [n_samples, n_features], [n_samples]
    ipdb.set_trace()
    clf = svm.LinearSVC().fit(X, Y)
    # Need to use clf.classes_, clf.coef_, clf.intercept_

    # this basically just forms the coefficients into weights, it's basically a reshape operation
    weights = np.zeros((len(feature_list), features_per_dim, n_classes))
    for i in feature_sliced:
        weights[
            i // features_per_dim, i % features_per_dim] = clf.coef_[:, i]
    intercept = clf.intercept_

    # run the test data with the SVM
    with test_model:
        # this may not work
        feed_net.d_f.__init__(testset[0], testset[1], t_len, dims, n_classes)

        # synapses for smoothing, because that's important according to Aaron's paper
        predict_tau = 0.005
        lowpass = 0.05
        tau_ratio = predic_tau / lowpass

        scores = nengo.Node(size_in=n_classes)

        predict = nengo.networks.EnsembleArray(n_neurons, n_classes, seed=SEED+1)
        nengo.Connection(scores, predict.input, transform=tau_ratio,
            synapse=predict_tau)
        nengo.Connection(predict.input, predict.output,
            transform=1-tau_ratio, synapse=predict_tau)

        for dim in range(dims):
            w = weights[dim]
            subset = (w != 0).any(axis=1)
            assert subset.shape == (features_per_dim,)

            nengo.Connection(feat_pops[dim].neurons, scores,
                transform=w[subset].T, synapse=None)

        bias = nengo.Node(output=[1], label="bias")
        nengo.Connection(bias, scores,
                 transform=intercept[:, None], synapse=None)

        p_out = nengo.Probe(predict.output)
        p_correct = nengo.Probe(feed_net.get_ans)

    print("test simulation start")
    sim_test = nengo.Simulator(test_model)
    with sim_test:
        sim_test.run((t_len + PAUSE)*testset[0].shape[0])
    print("test simulation start")

    # TODO: analyse the dataset
    #return get_accuracy(sim.data[p_out], sim.data[p_correct])
    # For now, just plot the results
    plt.plot(sim_test.trange(), nengo.Lowpass(0.01).filt(sim_test.data[p_out]), alpha=0.6)
    plt.plot(sim_test.trange(), sim_test.data[p_correct], alpha=0.6)
    #plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
    #plt.legend()
    plt.show()
    ipdb.set_trace()