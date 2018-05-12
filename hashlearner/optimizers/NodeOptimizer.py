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
import datetime


def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def log(row, ts):
    path = "results{}.csv".format(ts)
    with open(path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(row)


def get_convolution_set(convolve_range: tuple, convolution_density: int):
    convolution_list = list(itertools.combinations(range(convolve_range[0], convolve_range[1], convolution_density), 2))
    return convolution_list


def optimize_simple_nodes(mnist_data):
    ts = str(int(time.time()))
    train, test = sk_model.train_test_split(mnist_data, test_size=0.3)
    convolution_set = list(itertools.combinations(range(4, 15, 1), r=2))
    down_scale_ratios = array(list(range(2, 9, 1))) / 10
    binarize_threshold = list(range(150, 240, 10))
    parameter_list = list(itertools.product(convolution_set, binarize_threshold, down_scale_ratios))
    log(["success_rate", "convolve_shape_x","convolve_shape_y", "binarize_threshold_x", "binarize_threshold_y", "down_scale_ratio"], ts)

    indexs = list(range(2719, len(parameter_list)))
    for i in indexs:
        print("####### Testing Index: {} of {} `#######".format(str(i), str(len(parameter_list))))
        conv = parameter_list[i][0]
        thresh = parameter_list[i][1]
        down_scale = parameter_list[i][2]
        simple_node = SimpleHBaseMnistNode(setup_table=True, convolve_shape=conv, binarize_threshold=thresh, down_scale_ratio=down_scale)
        success_rate = get_prediction_rate(simple_node, train, test)
        row = [success_rate, conv[0],conv[1], thresh, down_scale]
        log(row, ts)


def get_prediction_rate(simple_mnist_node: SimpleHBaseMnistNode, train_mnist: list, test_mnist: list):
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
    print("Average Success Rate is: {0} for Node: {1}".format(str(success_rate), str(simple_mnist_node)))

    return success_rate


def main():
    t0 = time.time()
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:300]

    optimize_simple_nodes(mnist_data)

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
