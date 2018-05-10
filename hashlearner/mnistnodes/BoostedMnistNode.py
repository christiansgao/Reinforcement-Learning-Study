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

class BoostedMnistNode(MnistNode):
    '''
    Simple Hash node
    '''

    DEFAULT_SD_SHRINK = .1  # Amount of sd increments to smooth by
    CONVOLVE_SHAPE = (10, 10)
    DOWN_SCALE_RATIO = .5
    BINARIZE_THRESHOLD = 80
    TABLE_NAME = "L2Boost-Mnist-Node-1"
    BATCH_SIZE = 100
    POOL_SIZE = 10
    CONNECTION_POOL_SIZE = 300
    COLUMN_NAME = "number"

    def __init__(self, beta_weights = np.ones(10), setup_table=False, convolve_shape=CONVOLVE_SHAPE,
                 down_scale_ratio=DOWN_SCALE_RATIO, binarize_threshold=BINARIZE_THRESHOLD):
        super().__init__()

        self.binarize_threshold = binarize_threshold
        self.convolve_shape = convolve_shape
        self.down_scale_ratio = down_scale_ratio
        self.beta_weights = beta_weights
        if setup_table:
            self.setup()

        self.cached_predictions = []

    def train_batch(self, mnist_batch, index):
        '''
        :type mnist_batch: list of tuple
        :type deviate: boolean
        :rtype: None
        '''

        t0 = time.time()

        connection_pool = ConnectionPool(size=self.CONNECTION_POOL_SIZE, host=HBaseManager.HOST, port=HBaseManager.PORT)
        hbase_manager = HBaseManager(connection_pool)

        process_pool = Pool(self.POOL_SIZE)
        thread_pool = ThreadPool(self.POOL_SIZE)
        n = len(mnist_batch)

        numbers, mnist_images = MnistHelper.extract_numbers_images(mnist_batch)
        mnist_images = [mnist_obs[MnistModel.PREDICTOR_INDEX] for mnist_obs in mnist_batch]
        indexs = list(range(n))

        extract_process = process_pool.starmap_async(self.extract_keys, zip(mnist_images, indexs))
        extracted_keys = extract_process.get()

        store_hash_args = zip(extracted_keys, numbers, indexs)
        [self.store_hash_values(k, n, hbase_manager, i) for k, n, i in store_hash_args]

        process_pool.close()
        thread_pool.close()

        t1 = time.time()
        print("Time taken to train batch {} : {} Seconds".format(str(index),str(t1 - t0)))

    def extract_keys(self, mnist_image: np.ndarray, index: int):

        convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.convolve_shape)
        images_rescaled = MnistHelper.down_scale_images(convolved_images, ratio=self.down_scale_ratio)
        binarized_images = MnistHelper.binarize_images(images_rescaled, threshold=self.binarize_threshold)
        feature_positions = list(range(len(binarized_images)))

        hash_keys = [re.sub("[^0-9]+", "", str(binarized_image)) for binarized_image in binarized_images]
        feature_position_pair = list(zip(hash_keys, feature_positions))
        useful_features = list(filter(lambda x: int(x[0]) != 0, feature_position_pair))
        positioned_shaped_keys = [ "{}-{}-{}-{}".format(str(position),self.convolve_shape[0], self.convolve_shape[1],key)
                                   for key, position in useful_features]

        return positioned_shaped_keys

    '''def extract_keys(self, mnist_image: np.ndarray, index: int):
        convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.convolve_shape)
        images_rescaled = MnistHelper.down_scale_images(convolved_images, ratio=self.down_scale_ratio)
        binarized_images = MnistHelper.binarize_images(images_rescaled, threshold=self.binarize_threshold)
        feature_positions = list(range(len(binarized_images)))

        hash_keys = [re.sub("[^0-9]+", "", str(binarized_image)) for binarized_image in binarized_images]
        feature_position_pair = list(zip(hash_keys, feature_positions))
        useful_features = list(filter(lambda x: int(x[0]) != 0, feature_position_pair))
        positioned_keys = [str(position) + "-" + key for key, position in useful_features]

        return positioned_keys'''

    def predict_from_image_batch(self, mnist_batch, index):

        t0 = time.time()
        connection_pool = ConnectionPool(size=self.CONNECTION_POOL_SIZE, host=HBaseManager.HOST, port=HBaseManager.PORT)
        hbase_manager = HBaseManager(connection_pool)

        process_pool = Pool(self.POOL_SIZE)
        n = len(mnist_batch)

        indexs = list(range(n))

        extract_process = process_pool.starmap_async(self.extract_keys, zip(mnist_batch, indexs))
        extracted_keys = extract_process.get()

        predict_hash_args = zip(extracted_keys, indexs)

        predictions = [self.predict_hash_values(keys, hbase_manager, i) for keys, i in predict_hash_args]

        process_pool.close()

        t1 = time.time()
        print("Mnist Batch {} predicted in: {} Seconds".format(str(index), str(t1 - t0)))

        return predictions


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:100]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.2)

    mnist_node = BoostedMnistNode(setup_table=True, convolve_shape= (10, 10))
    status = mnist_node.train_node(train)
    print("training status: " + str(status))

    true_numbers = [image[0] for image in test]
    test_images = [image[1] for image in test]
    print("Starting predictions")

    predictions = mnist_node.predict_from_images(test_images)

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
