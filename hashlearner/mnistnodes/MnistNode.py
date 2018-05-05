from abc import ABC, abstractmethod
from hbase.HBaseManager import HBaseRow, HBaseManager
from hashlearner.helper.mnist import MnistHelper
from happybase import ConnectionPool

import itertools
import random

class MnistNode(ABC):

    DEFAULT_DECIMAL_PRECISION = 1 # precision to 1 decimal point

    '''
    Generic Mnist Node takes in predictor indexs and response indexs.
    '''

    def __init__(self, decimal_precision = DEFAULT_DECIMAL_PRECISION):
        '''
        :type hashtype: int
        :type predictor_indexs: list
        :type response_index: int
        '''
        self.decimal_precision = decimal_precision # all results get rounded to this precision
        super().__init__()

    @abstractmethod
    def train_node(self, mnist_data):
        pass

    @abstractmethod
    def predict_from_images(self, key):
        pass

    def store_hash_values(self, hash_keys: list, number: int, hbase_manager: HBaseManager, index: int):
        batch_insert_rows = [
            HBaseRow(row_key=hash_key, row_values={self.COLUMN_NAME: number}, family_name=HBaseManager.FAMILY_NAME)
            for hash_key in hash_keys
        ]
        status = hbase_manager.batch_insert(self.TABLE_NAME, batch_insert_rows)
        return True

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

    def setup(self):
        HBaseManager(ConnectionPool(size=1, host=HBaseManager.HOST, port=HBaseManager.PORT)).create_table(
            table_name=self.TABLE_NAME, delete=True)

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