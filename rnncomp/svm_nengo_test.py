# based off of Aaron's SVM network in http://compneuro.uwaterloo.ca/publications/voelker2016a.html

from constants import *
from dataman import create_feed_net

import nengo
from nengolib.synapses import HeteroSynapse, Bandpass
from sklearn import svm
import numpy as np


def multisynapse(src, dest, sub_features):

    synapses = [Bandpass(freq, Q) for (freq, Q) in sub_features]
    syn = nengo.Node(size_in=1, output=HeteroSynapse(synapses, dt=dt))

    nengo.Connection(src, syn, synapse=None)
    nengo.Connection(
        syn, dest.neurons, synapse=None,
        function=lambda x, transform=np.squeeze(dest.encoders): transform*x)


def svm_freq(t_len, dims, n_classes, alif=False):

    tau = 0.1
    # setup the feature craziness
    freq_range = (0, 500)
    Q_range = (2, 50)
    features_per_dim = 400
    feat_pops = []
    feat_list = []
    enc_list = []

    def train(dataset, corset):

        # make a model and run it to get spiking data
        # as acquired when passed through multiple bandpass filters
        train_model = nengo.Network()
        with train_model:
            feed_net = create_feed_net(dataset, corset, t_len, dims, n_classes)

            if alif:
                state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED, neuron_type=nengo.AdaptiveLIF())
            else:
                state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED)

            nengo.Connection(feed_net.q_in, state.input, synapse=None)

            for dim in range(dims):
                # this declaration needed for the multisynapse transform
                encoders = nengo.dists.UniformHypersphere(
                            surface=True).sample(features_per_dim, 1)
                enc_list.append(encoders)
                feat_pop = nengo.Ensemble(features_per_dim, 1, encoders=encoders, seed=SEED+dim)
                feat_pops.append(feat_pop)

                sub_features = zip(
                    nengo.dists.Uniform(*freq_range).sample(features_per_dim),
                    nengo.dists.Uniform(*Q_range).sample(features_per_dim)
                )
                feat_list.append(sub_features)
                multisynapse(state.ensembles[dim], feat_pop, sub_features)

            p_target = nengo.Probe(feed_net.get_ans, sample_every=sample_every, synapse=None)

            p_features = [
                nengo.Probe(
                    feat_pop.neurons, sample_every=sample_every, synapse=tau)
                for feat_pop in feat_pops]

        print("training simulation start")
        sim_train = nengo.Simulator(train_model)
        with sim_train:
            sim_train.run((t_len + PAUSE)*dataset.shape[0])
        print("training simulation done")

        # TODO: Enable logging and close the files here

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
        # Need to use clf.coef_, clf.intercept_

        # this basically just forms the coefficients into weights, it's basically a reshape operation
        weights = np.zeros((len(feat_list), features_per_dim, n_classes))
        for i in xrange(feature_num):
            weights[i // features_per_dim, i % features_per_dim] = clf.coef_[:, i]

        return weights, clf.intercept_

    def test(weights, intercept, testset=None):
        # run the test data with the SVM

        test_model = nengo.Network()
        with test_model:
            feed_net = create_feed_net(testset[0], testset[1], t_len, dims, n_classes)

            if alif:
                state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED, neuron_type=nengo.AdaptiveLIF())
            else:
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
                feat_pop = nengo.Ensemble(features_per_dim, 1, encoders=enc_list[dim],
                                          seed=SEED+dim)
                multisynapse(state.ensembles[dim], feat_pop, feat_list[dim])

                nengo.Connection(feat_pop.neurons, scores,
                                 transform=weights[dim].T, synapse=None)

            bias = nengo.Node(output=[1], label="bias")
            nengo.Connection(bias, scores,
                             transform=intercept[:, None], synapse=None)

            p_out = nengo.Probe(predict.output, sample_every=sample_every)
            p_correct = nengo.Probe(feed_net.get_ans, sample_every=sample_every)

        print("test simulation start")
        sim_test = nengo.Simulator(test_model)
        with sim_test:
            sim_test.run((t_len + PAUSE)*testset[0].shape[0])
        print("test simulation done")

        # TODO: Enable logging and close the files here

        return sim_test.data[p_out], sim_test.data[p_correct]

    return train, test
