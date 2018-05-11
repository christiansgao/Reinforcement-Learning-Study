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
from hashlearner.helper import CSVHelper
from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hashlearner.mnistnodes.BoostedMnistNode import BoostedMnistNode
from hashlearner.mnistnodes.FilterHBaseMnistNode import FilterHBaseMnistNode

from random import shuffle
import sklearn.ensemble.gradient_boosting
import sys


class BoostedMnistModel(MnistModel):
    def __init__(self, mnist_data, optimization_len=100, train_test_ratio=.8):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''
        super().__init__()

        self.initializes_model()
        self.optimization_len = optimization_len
        self.train_test_ratio = train_test_ratio
        self.mnist_data = mnist_data
        self.boosted_node_list = []

    def initializes_model(self):

        self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(10, 10), binarize_threshold=80, down_scale_ratio=.5))
        self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(5, 9), binarize_threshold=160, down_scale_ratio=.6))
        self.mnist_node_list.append(FilterHBaseMnistNode())

        self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(8, 13), binarize_threshold=200, down_scale_ratio=.4))
        self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(4, 6), binarize_threshold=200, down_scale_ratio=.8))
        self.mnist_node_list.append(BoostedMnistNode(convolve_shape=(9, 10), binarize_threshold=210, down_scale_ratio=.6))

    def train_model(self, iterations=1):

        # train, test = sk_model.train_test_split(self.mnist_data, test_size=0.2)
        split = int(len(self.mnist_data)*.5)
        train = self.mnist_data[:split]
        test = self.mnist_data[split:]

        self.train_nodes(train)
        # shuffle(self.mnist_node_list)

        self.initial_benchmark(test)
        self._boost_nodes(test)

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

        print("expected numbers: \n" + str(expected))

        best_loss = sys.float_info.max

        for mnist_node in self.mnist_node_list:
            predictions = self.predict_from_images(mnist_images)
            loss = self.get_loss(predictions, expected)

            if loss < best_loss:
                best_loss = loss

            print("Best Loss for: {} is {}!!!".format(str(mnist_node), str(loss)))
            print("predictions for node {}: \n{}".format(str(mnist_node), str(predictions)))

        print("Best Loss for Nodes is {}!!!".format(str(best_loss)))

    def _boost_nodes(self, test_mnist_data):

        print("####### Optimizing and Boosting #######")

        expected, mnist_images = MnistHelper.extract_numbers_images(test_mnist_data)

        for mnist_node_index in range(0, len(self.mnist_node_list)):  # type: BoostedMnistNode
            mnist_node = self.mnist_node_list[mnist_node_index]
            self.set_node_weights(expected, mnist_node, mnist_images)

    def set_node_weights(self, expected, mnist_node, mnist_images):
        '''
        :type expected:
        :type mnist_node:
        :type mnist_images:
        :rtype:
        '''
        predictions = self.predict_from_images(mnist_images)

        for i in range(0, 10):
            mnist_node.beta_weights[i] = 1 - self.get_loss(predictions, expected, i)

        new_predictions = self.predict_from_images(mnist_images)
        loss = self.get_loss(new_predictions, expected, i)
        #print("Weights: {}".format(str(mnist_node.beta_weights)))
        #print("predictions for model: \n{}".format(str(predictions)))
        print("New Loss After setting weights: " + str(loss))

    def get_loss(self, predictions, expected, i=None):

        if i:
            indexes = np.where(np.array(predictions) == str(i))[0]
            predictions = [predictions[i] for i in indexes]
            expected = [expected[i] for i in indexes]

        if len(predictions) == 0:
            return .5

        confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)
        correct_classifications = np.diagonal(confusion_matrix);
        loss = 1 - sum(correct_classifications) / np.sum(confusion_matrix)
        return loss

    def predict_from_images(self, images, use_cache=True):
        '''
        :type images: pandas.DataFrame
        :rtype: list
        '''

        final_predictions = []
        candidates = [{} for _ in range(len(images))]

        for mnist_node in self.mnist_node_list:  # type: BoostedMnistNode
            predictions = mnist_node.predict_from_images(images, use_cache=use_cache)

            for prediction, candidate in zip(predictions, candidates):
                if not prediction in candidate:
                    candidate[prediction] = mnist_node.beta_weights[int(prediction)]
                else:
                    candidate[prediction] += mnist_node.beta_weights[int(prediction)]

        for candidate in candidates:
            final_prediction = max(candidate, key=candidate.get) if len(candidates) != 0 else np.random.choice(
                self.response_set)
            final_predictions.append(final_prediction)

        #print("final candidates: " + str(candidates))

        return final_predictions


def main():
    mnist_data = MnistLoader.read_mnist()

    t0 = time.time()

    # train, test = sk_model.train_test_split(mnist_data, test_size=0.1)
    train = mnist_data[:2000]
    test = mnist_data[2000:3000]

    mnist_model = BoostedMnistModel(mnist_data=train)
    mnist_model.train_model(iterations=1)

    expected, test_images = MnistHelper.extract_numbers_images(test)

    print("Starting predictions")

    predictions = mnist_model.predict_from_images(test_images, use_cache=False)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)

    CSVHelper.write_predictions(expected, predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    print("true numbers: " + str(expected))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
