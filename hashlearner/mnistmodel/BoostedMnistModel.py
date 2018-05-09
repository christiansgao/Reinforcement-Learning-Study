import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model
import array
import itertools
from hashlearner.helper.mnist import LossFunctions
import copy

import scipy.optimize.linesearch

from hashlearner.mnistmodel.MnistModel import MnistModel
from hashlearner.mnistnodes.MnistNode import MnistNode
from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hashlearner.mnistnodes.BoostedMnistNode import BoostedMnistNode
from random import shuffle
import sklearn.ensemble.gradient_boosting
import sys


class BoostedMnistModel(MnistModel):
    def __init__(self, iterations, mnist_data, optimization_len=100, train_test_ratio=.8):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''
        super().__init__()

        self.initializes_model()
        self.iterations = iterations
        self.optimization_len = optimization_len
        self.train_test_ratio = train_test_ratio
        self.mnist_data = mnist_data
        self.boosted_node_list = []

    def initializes_model(self):

        self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(5, 9), binarize_threshold=160, down_scale_ratio=.6))
        # self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(8, 13), binarize_threshold=200, down_scale_ratio=.4))
        # self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(4, 6), binarize_threshold=200, down_scale_ratio=.8))

    def train_model(self):

        #train, test = sk_model.train_test_split(self.mnist_data, test_size=0.2)
        train = self.mnist_data

        self.train_nodes(train)
        # shuffle(self.mnist_node_list)

        # self.initial_benchmark(test_mnist_data)
        #self._boost_node(test)

    def train_nodes(self, mnist_data):

        BoostedMnistNode().setup()

        index = 1
        trained_node_list = []
        for hash_node in self.mnist_node_list:  # type: MnistNode
            print("####### Training Node Number: {} #######".format(str(index)))
            hash_node.train_node(mnist_data)
            index += 1

    def initial_benchmark(self, test_mnist_data):

        print("\n####### Initial Benchmarks #######\n")

        expected, mnist_images = MnistHelper.extract_numbers_images(test_mnist_data)
        best_loss = sys.float_info.max

        print("expected numbers: \n" + str(expected))

        for mnist_node in self.mnist_node_list:
            self.boosted_node_list = [mnist_node]
            loss, predictions = self.calculate_loss(expected, mnist_images)
            if loss < best_loss:
                best_loss = loss
            print("Loss for: {} is {}!!!".format(str(mnist_node), str(loss)))
            print("predictions for node {}: \n{}".format(str(mnist_node), str(predictions)))

        print("Best Loss for Nodes is {}!!!".format(str(best_loss)))

    def _boost_node(self, test_mnist_data):

        self.boosted_node_list = []
        print("####### Optimizing and Boosting #######")

        expected, mnist_images = MnistHelper.extract_numbers_images(test_mnist_data)

        for mnist_node_index in range(0, len(self.mnist_node_list)):  # type: BoostedMnistNode
            mnist_node = self.mnist_node_list[mnist_node_index]
            self.boosted_node_list.append(mnist_node)
            if len(self.boosted_node_list) == 1:
                continue

            self.iterate_weights(expected, mnist_node, mnist_images)

    def iterate_weights(self, expected, mnist_node, mnist_images):
        '''
        :type expected:
        :type mnist_node:
        :type mnist_images:
        :rtype:
        '''
        # test_weights = np.arange(0.0, 2.1, 0.1)
        test_weights = [1]
        best_weights = np.ones(10)
        best_loss = sys.float_info.max
        mnist_node.beta_weights = np.ones(10)

        for i in range(0, len(mnist_node.beta_weights)):
            for t in test_weights:
                mnist_node.beta_weights[i] = t
                loss, predictions = self.calculate_loss(expected, mnist_images)
                if loss < best_loss:
                    best_weights = np.array(mnist_node.beta_weights)
                    best_loss = loss
                mnist_node.beta_weights = np.array(best_weights)
                print("Current Loss: {}".format(str(loss)))
                print("Weights: {}".format(str(mnist_node.beta_weights)))
                print("predictions for model: \n{}".format(str(predictions)))

        print("Best For Model Loss: " + str(best_loss))

    def predict_from_images(self, images):
        '''
        :type images: pandas.DataFrame
        :rtype: list
        '''

        final_predictions = []
        candidates = [{} for _ in range(len(images))]

        for boosted_node in self.boosted_node_list:  # type: BoostedMnistNode
            predictions = boosted_node.predict_from_images(images)

            for prediction, candidate in zip(predictions, candidates):
                if not prediction in candidate:
                    candidate[prediction] = boosted_node.beta_weights[int(prediction)]
                else:
                    candidate[prediction] += boosted_node.beta_weights[int(prediction)]

        for candidate in candidates:
            final_prediction = max(candidate, key=candidate.get) if len(candidates) != 0 else np.random.choice(
                self.response_set)
            final_predictions.append(final_prediction)

        print("final candidates: " + str(candidates))

        return final_predictions

    def calculate_loss(self, expected, mnist_images):
        predictions = self.predict_from_images(mnist_images)
        loss = LossFunctions.zero_one_loss(expected, predictions)
        return loss, predictions


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:1100]

    t0 = time.time()

    # train, test = sk_model.train_test_split(mnist_data, test_size=0.1)
    train = mnist_data[:10]
    test = mnist_data[90:100]
    test2 = mnist_data[100:110]

    mnist_model = BoostedMnistModel(iterations=1, mnist_data=mnist_data)
    mnist_model.train_model(train, test)

    true_numbers, test_images = MnistHelper.extract_numbers_images(test2)

    print("Starting predictions")

    predictions = mnist_model.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_numbers, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    print("true numbers: " + str(true_numbers))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))
    print("Total Loss is: " + str(mnist_model.calculate_loss(true_numbers, test_images)[0]))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
