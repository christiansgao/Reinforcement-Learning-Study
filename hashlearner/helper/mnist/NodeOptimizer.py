import itertools
import random
import re
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model
from happybase import ConnectionPool

from hashlearner.mnistnodes.MnistNode import MnistNode
from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hbase.HBaseManager import HBaseManager, HBaseRow
from hashlearner.mnistmodel.MnistModel import MnistModel
from hashlearner.mnistnodes.SimpleHBaseMnistNode import SimpleHBaseMnistNode
import csv
import itertools
from numpy import array


def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

def log():
    pass

def get_convolution_set(convolve_range: tuple, convolution_density: int):
    convolution_list = list(itertools.combinations(range(convolve_range[0], convolve_range[1],convolution_density),2))
    return convolution_list


def optimize_simple_nodes(mnist_data):

    train, test = sk_model.train_test_split(mnist_data, test_size=0.2)
    convolution_set =list(itertools.combinations(np.linspace(1,28,1),r=2))
    down_scale_ratios = array(list(range(1,10,1)))/10
    binarize_threshold = list(range(10,250,10))
    parameter_list = list(itertools.product(convolution_set, binarize_threshold))
    pass


def get_prediction_rate(simple_mnist_node: SimpleHBaseMnistNode, train_mnist:list, test_mnist: list):
    mnist_node = simple_mnist_node
    status = mnist_node.train_node(train_mnist)
    print("training status: " + str(status))

    true_numbers = [image[0] for image in test_mnist]
    test_images = [image[1] for image in test_mnist]
    print("Starting predictions")

    predictions = mnist_node.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_numbers, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    print("true numbers: " + str(true_numbers))
    print("predictions: " + str(predictions))
    print("Average Success Rate is: {} for Node: {}".format(str(success_rate),str(simple_mnist_node)))

    return predictions

def main():

    t0 = time.time()
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:100]

    optimize_simple_nodes(mnist_data)

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
