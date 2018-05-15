
import time

import numpy as np
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model

import hashlearner.helper.CSVHelper as CSVHelper
from hashlearner.helper.mnist import MnistLoader
from hashlearner.mnistnodes.SimpleHBaseMnistNode import SimpleHBaseMnistNode
from string import Template
from hashlearner.mnistnodes.FilterHBaseMnistNode import FilterHBaseMnistNode
from hashlearner.mnistnodes.MnistNode import MnistNode


def execute_iterations():

    mnist_data = MnistLoader.read_mnist()
    train = mnist_data[:50000]
    test = mnist_data[50000:]

    nodes = [FilterHBaseMnistNode(convolve_shape=(8, 13), binarize_threshold=200, down_scale_ratio=.4),
             FilterHBaseMnistNode(down_scale_ratio=.4),
             FilterHBaseMnistNode(convolve_shape=(5, 9), binarize_threshold=160, down_scale_ratio=.6),
             FilterHBaseMnistNode(convolve_shape=(4, 6), binarize_threshold=200, down_scale_ratio=.8),
             FilterHBaseMnistNode(convolve_shape=(12, 6), binarize_threshold=150, down_scale_ratio=.5)
             ]

    for node in nodes:
        run_node(node, train, test)

def extract_name(node: SimpleHBaseMnistNode):
    name_template = Template("${name}_${conv_1}.${conv_2}_${thresh}_${scale}")
    name = name_template.substitute(name= node.table_name, conv_1 = str(node.convolve_shape[0]), conv_2=str(node.convolve_shape[1]), thresh=node.binarize_threshold, scale=node.down_scale_ratio)
    return name

def run_node(mnist_node: SimpleHBaseMnistNode, train, test):
    name = extract_name(mnist_node)
    print("starting hbase mnist node" + name)
    t0 = time.time()

    status = mnist_node.train_node(train)
    print("training status: " + str(status))

    expected = [image[0] for image in test]
    test_images = [image[1] for image in test]
    print("Starting predictions")

    predictions = mnist_node.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    CSVHelper.write_predictions(expected, predictions, name=name)

    print("true numbers: " + str(expected))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")

def main():
    execute_iterations()

if __name__ == "__main__":
    main()
