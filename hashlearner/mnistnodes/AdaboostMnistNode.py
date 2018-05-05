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


class AdaboostMnistNode(MnistNode):
    '''
    Simple Hash node
    '''

    DEFAULT_SD_SHRINK = .1  # Amount of sd increments to smooth by
    DEFAULT_SMOOTH_SPREAD = list(range(-5, 5, 1))  # How many sd on each side
    CONVOLVE_SHAPE = (10, 10)
    DOWN_SCALE_RATIO = .5
    BINARIZE_THRESHOLD = 80
    ALL_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    TABLE_NAME = "Simple-Mnist-Node-1"
    BATCH_SIZE = 1000
    POOL_SIZE = 10
    CONNECTION_POOL_SIZE = 300
    COLUMN_NAME = "number"

    def __init__(self, alpha_weight = 1, setup_table=False, convolve_shape=CONVOLVE_SHAPE,
                 down_scale_ratio=DOWN_SCALE_RATIO, binarize_threshold=BINARIZE_THRESHOLD):
        super().__init__()

        self.binarize_threshold = binarize_threshold
        self.convolve_shape = convolve_shape
        self.down_scale_ratio = down_scale_ratio
        self.beta_weight = alpha_weight
        if setup_table:
            self.setup()

        self.cached_predictions = []

    def setup(self):
        HBaseManager(ConnectionPool(size=1, host=HBaseManager.HOST, port=HBaseManager.PORT)).create_table(
            table_name=self.TABLE_NAME, delete=True)

    def train_batch(self, mnist_batch, index):
        '''
        :type mnist_batch: list of tuple
        :type deviate: boolean
        :rtype: None
        '''
        print("Training Batch {}".format(str(index)))

        t0 = time.time()

        connection_pool = ConnectionPool(size=self.CONNECTION_POOL_SIZE, host=HBaseManager.HOST, port=HBaseManager.PORT)
        hbase_manager = HBaseManager(connection_pool)

        process_pool = Pool(self.POOL_SIZE)
        thread_pool = ThreadPool(self.POOL_SIZE)
        n = len(mnist_batch)

        numbers = [mnist_obs[MnistModel.RESPONSE_INDEX] for mnist_obs in mnist_batch]
        mnist_images = [mnist_obs[MnistModel.PREDICTOR_INDEX] for mnist_obs in mnist_batch]
        indexs = list(range(n))

        extract_process = process_pool.starmap_async(self.extract_keys, zip(mnist_images, indexs))
        extracted_keys = extract_process.get()

        store_hash_args = zip(extracted_keys, numbers, indexs)
        [self.store_hash_values(k, n, hbase_manager, i) for k, n, i in store_hash_args]

        process_pool.close()
        thread_pool.close()

        t1 = time.time()
        print("Time taken to train batch: " + str(t1 - t0) + " Seconds")

        print("Training for Mnist Batch Finished")

    def train_node(self, mnist_data):
        '''
        :type mnist_data: list of tuple
        :type deviate: boolean
        :rtype: None
        '''

        batches = MnistHelper.batch(mnist_data, self.BATCH_SIZE)
        indexs = list(range(len(batches)))

        [self.train_batch(batch, index) for batch, index in zip(batches, indexs)]

        print("Training for Mnist Data Finished")

        return True

    def store_hash_values(self, hash_keys: list, number: int, hbase_manager: HBaseManager, index: int):
        batch_insert_rows = [
            HBaseRow(row_key=hash_key, row_values={self.COLUMN_NAME: number}, family_name=HBaseManager.FAMILY_NAME)
            for hash_key in hash_keys
        ]
        status = hbase_manager.batch_insert(self.TABLE_NAME, batch_insert_rows)
        # print("trained keys for image {0} with status: {1} ".format(str(index), str(status)))
        return True

    def extract_keys(self, mnist_image: np.ndarray, index: int):
        convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.convolve_shape)
        images_rescaled = MnistHelper.down_scale_images(convolved_images, ratio=self.down_scale_ratio)
        binarized_images = MnistHelper.binarize_images(images_rescaled, threshold=self.binarize_threshold)
        feature_positions = list(range(len(binarized_images)))

        hash_keys = [re.sub("[^0-9]+", "", str(binarized_image)) for binarized_image in binarized_images]
        feature_position_pair = list(zip(hash_keys, feature_positions))
        useful_features = list(filter(lambda x: int(x[0]) != 0, feature_position_pair))
        positioned_keys = [str(position) + "-" + key for key, position in useful_features]

        # print("extracted key from image: " + str(index))
        return positioned_keys

    def predict_from_images(self, mnist_data):
        print("Starting Predicting for Mnist data")

        if self.cached_predictions:
            return self.cached_predictions

        batches = MnistHelper.batch(mnist_data, self.BATCH_SIZE)

        indexs = list(range(len(batches)))
        predictions = [self.predict_from_image_batch(batch, index) for batch, index in zip(batches, indexs)]
        flat_predictions = list(itertools.chain.from_iterable(predictions))

        print("Predicting for Mnist Data Finished")

        self.cached_predictions = flat_predictions

        return flat_predictions

    def predict_hash_values(self, hash_keys: list, hbase_manager: HBaseManager, index):
        # print("predicting image: " + str(index))

        if len(hash_keys) == 0:
            print("no good hash keys")
            return random.choice(self.ALL_DIGITS)

        hash_rows = hbase_manager.batch_get_rows(self.TABLE_NAME, hash_keys)
        hashed_predictions = [hash_row.row_values[self.COLUMN_NAME] for hash_row in hash_rows]
        if len(hashed_predictions) == 0:
            print("no collision predictions")
            return random.choice(self.ALL_DIGITS)

        best_prediction = max(hashed_predictions, key=hashed_predictions.count)

        return best_prediction

    def predict_from_image_batch(self, mnist_batch, index):

        print("Predicting Batch {}".format(str(index)))

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

    def __str__(self):
        return self.predictions_map.__str__()


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:200]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.2)

    mnist_node = AdaboostMnistNode(setup_table=True, convolve_shape= (10,10))
    status = mnist_node.train_node(train)
    print("training status: " + str(status))

    true_numbers = [image[0] for image in test]
    test_images = [image[1] for image in test]
    print("Starting predictions")

    predictions = mnist_node.predict_from_images(test_images)
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
