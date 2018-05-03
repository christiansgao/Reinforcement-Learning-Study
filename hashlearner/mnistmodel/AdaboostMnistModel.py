import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model
import array
import itertools

from hashlearner.mnistmodel.MnistModel import MnistModel
from hashlearner.mnistnodes.MnistNode import MnistNode
from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hashlearner.mnistnodes.SimpleHBaseMnistNode import SimpleHBaseMnistNode


class HBaseMnistModel(MnistModel):
    def __init__(self, iterations, optimization_len=100, train_test_ratio=.8):
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

    def initializes_model(self):

        # SimpleHBaseMnistNode().setup()

        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="1", convolve_shape=(6, 6), binarize_threshold=80))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="2", convolve_shape=(7, 7), binarize_threshold=80))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="3", convolve_shape=(8, 8), binarize_threshold=80))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="4", convolve_shape=(9, 9), binarize_threshold=80))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="5", convolve_shape=(10, 10), binarize_threshold=80))

        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="6", convolve_shape=(6, 6), binarize_threshold=100))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="7", convolve_shape=(7, 7), binarize_threshold=100))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="8", convolve_shape=(8, 8), binarize_threshold=100))
        self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="9", convolve_shape=(9, 9), binarize_threshold=100))
        self.mnist_node_list.append(
            SimpleHBaseMnistNode(node_name="10", convolve_shape=(10, 10), binarize_threshold=100))

        self.mnist_node_list.append(
            SimpleHBaseMnistNode(node_name="11", convolve_shape=(6, 6), binarize_threshold=90, down_scale_ratio=.7))
        self.mnist_node_list.append(
            SimpleHBaseMnistNode(node_name="12", convolve_shape=(7, 7), binarize_threshold=90, down_scale_ratio=.7))
        # self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="13",convolve_shape=(8, 8), binarize_threshold=90, down_scale_ratio= .7))
        # self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="14",convolve_shape=(9, 9), binarize_threshold=90, down_scale_ratio= .7))
        # self.mnist_node_list.append(SimpleHBaseMnistNode(node_name="15",convolve_shape=(10, 10), binarize_threshold=90, down_scale_ratio= .7))

    def train_model(self, mnist_data):

        index = 1
        trained_node_list = []
        for hash_node in self.mnist_node_list:  # type: MnistNode
            print("####### Training Node Number: {} #######".format(str(index)))
            hash_node.train_node(mnist_data)
            index += 1

    def exponential_loss(self, predictions: list, trained_node_list: list):
        trained_node_list

    def predict_from_images(self, images):
        '''
        :type images: pandas.DataFrame
        :rtype: list
        '''

        final_predictions = []
        candidates = [{} for _ in range(len(images))]

        for mnistNode in self.mnist_node_list:  # type: MnistNode
            print("########## Extracting Predictions for Node: {} ##########".format(mnistNode.node_name))
            predictions = mnistNode.predict_from_images(images)

            for prediction, candidate in zip(predictions, candidates):
                if not prediction in candidate:
                    candidate[prediction] = 1
                else:
                    candidate[prediction] += 1

        for candidate in candidates:
            final_prediction = max(candidate, key=candidate.get) if len(candidates) != 0 else np.random.choice(
                self.response_set)
            final_predictions.append(final_prediction)

        return final_predictions


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:10000]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.1)
    train = mnist_data[:9000]
    test = mnist_data[9000:]

    mnist_model = HBaseMnistModel()
    # status = mnist_model.train_model(train)
    # print("training status: " + str(status))

    true_numbers = [image[MnistModel.RESPONSE_INDEX] for image in test]
    test_images = [image[MnistModel.PREDICTOR_INDEX] for image in test]
    print("Starting predictions")

    predictions = mnist_model.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_numbers, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    print("true numbers: " + str(true_numbers))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
