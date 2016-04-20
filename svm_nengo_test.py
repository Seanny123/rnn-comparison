# based off of Aaron's SVM network in http://compneuro.uwaterloo.ca/publications/voelker2016a.html

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
    enc_list = []
    sample_every = 0.005

    # make a model and run it to get spiking data
    # as acquired when passed through multiple bandpass filters
    train_model = nengo.Network()
    with train_model:
        feed_net = create_feed_net(dataset[0], dataset[1], t_len, dims, n_classes)

        state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED)

        nengo.Connection(feed_net.q_in, state.input, synapse=None)


        for dim in range(dims):
            # this declaration needed for the multisynapse transform
            encoders = nengo.dists.UniformHypersphere(
                        surface=True).sample(features_per_dim, 1)
            enc_list.append(encoders)
            feat_pop = nengo.Ensemble(features_per_dim, 1, encoders=encoders,
                seed=SEED+dim)
            feat_pops.append(feat_pop)

            sub_features = zip(
                nengo.dists.Uniform(*freq_range).sample(features_per_dim),
                nengo.dists.Uniform(*Q_range).sample(features_per_dim)
            )
            feat_list.append(sub_features)
            multisynapse(state.ensembles[dim], feat_pop, sub_features)

        p_sig = nengo.Probe(feed_net.q_in, sample_every=sample_every, synapse=None)
        p_target = nengo.Probe(feed_net.get_ans, sample_every=sample_every, synapse=None)

        p_normal = nengo.Probe(state.output, sample_every=sample_every, synapse=tau)

        p_features = [
            nengo.Probe(
                feat_pop.neurons, sample_every=sample_every, synapse=tau)
            for feat_pop in feat_pops]

    print("training simulation start")
    sim_train = nengo.Simulator(train_model)
    with sim_train:
        sim_train.run((t_len + PAUSE)*dataset[0].shape[0])
    print("training simulation done")

    # pass the feature data and the target to train an SVM
    print("Training SVM")

    feature_num = features_per_dim*dims

    # decode the labels into 1D (should be done in numpy)
    yv = []
    for tar_val in sim_train.data[p_target]:
        if np.any(tar_val == 1):
            yv.append(np.argmax(tar_val))
        else:
            yv.append(0)
    yv = np.array(yv)

    xv = np.zeros((yv.shape[0], feature_num))
    for p_i, p_feat in enumerate(p_features):
        xv[:, p_i*400:(p_i+1)*400] = sim_train.data[p_feat]

    clf = svm.LinearSVC().fit(xv, yv)
    # Need to use clf.classes_, clf.coef_, clf.intercept_

    # this basically just forms the coefficients into weights, it's basically a reshape operation
    weights = np.zeros((len(feat_list), features_per_dim, n_classes))
    for i in xrange(feature_num):
        weights[
            i // features_per_dim, i % features_per_dim] = clf.coef_[:, i]
    intercept = clf.intercept_


    # run the test data with the SVM
    test_model = nengo.Network()
    with test_model:
        feed_net = create_feed_net(testset[0], testset[1], t_len, dims, n_classes)

        state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED)

        nengo.Connection(feed_net.q_in, state.input, synapse=None)

        # synapses for smoothing, because that's important according to Aaron's paper
        predict_tau = 0.005
        lowpass = 0.05
        tau_ratio = predict_tau / lowpass

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
            feat_pop = nengo.Ensemble(features_per_dim, 1, encoders=enc_list[dim],
                seed=SEED+dim)
            multisynapse(state.ensembles[dim], feat_pop, feat_list[dim])

            nengo.Connection(feat_pop.neurons, scores,
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
    print("test simulation done")

    # TODO: analyse the dataset
    #return get_accuracy(sim.data[p_out], sim.data[p_correct])
    # For now, just plot the results
    plt.plot(sim_test.trange(), nengo.Lowpass(0.01).filt(sim_test.data[p_out]), alpha=0.6)
    plt.plot(sim_test.trange(), sim_test.data[p_correct], alpha=0.6)
    #plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
    #plt.legend()
    plt.show()
    ipdb.set_trace()