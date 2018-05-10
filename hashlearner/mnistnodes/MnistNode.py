from abc import ABC, abstractmethod
from hbase.HBaseManager import HBaseRow, HBaseManager
from hashlearner.helper.mnist import MnistHelper
from happybase import ConnectionPool
from collections import Counter
import itertools
import random


class MnistNode(ABC):
    DEFAULT_DECIMAL_PRECISION = 1  # precision to 1 decimal point
    ALL_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    DEFAULT_PROB = .1

    '''
    Generic Mnist Node takes in predictor indexs and response indexs.
    '''

    def __init__(self, decimal_precision=DEFAULT_DECIMAL_PRECISION):
        '''
        :type hashtype: int
        :type predictor_indexs: list
        :type response_index: int
        '''
        self.decimal_precision = decimal_precision  # all results get rounded to this precision
        super().__init__()

    def __str__(self):
        return "Convolve Shape: {}, Down Scale Ratio: {}, Binarize Threshold: {}".format(self.convolve_shape,
                                                                                         self.down_scale_ratio,
                                                                                         self.binarize_threshold)

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

        if self.cached_predictions:
            return self.cached_predictions

        batches = MnistHelper.batch(mnist_data, self.BATCH_SIZE)

        indexs = list(range(len(batches)))
        predictions = [self.predict_from_image_batch(batch, index) for batch, index in
                                      zip(batches, indexs)]
        flat_predictions = list(itertools.chain.from_iterable(predictions))

        self.cached_predictions = flat_predictions

        return self.cached_predictions

    def get_hbase_hash_values(self, hash_keys: list, hbase_manager: HBaseManager, index):
        hash_rows = hbase_manager.batch_get_rows(self.TABLE_NAME, hash_keys)
        hashed_predictions = [hash_row.row_values[self.COLUMN_NAME] for hash_row in hash_rows]

    def predict_hash_values(self, hash_keys: list, hbase_manager: HBaseManager, index):

        if len(hash_keys) == 0:
            print("no good hash keys")
            return random.choice(self.ALL_DIGITS), self.get_default_probs()

        hash_rows = hbase_manager.batch_get_rows(self.TABLE_NAME, hash_keys)
        hashed_predictions = [hash_row.row_values[self.COLUMN_NAME] for hash_row in hash_rows]

        if len(hashed_predictions) == 0:
            print("no collision predictions")
            return random.choice(self.ALL_DIGITS), self.get_default_probs()

        test = dict((x,hashed_predictions.count(x)) for x in set(hashed_predictions))

        best_prediction = max(hashed_predictions, key=hashed_predictions.count)

        return best_prediction

    def predict_hash_values(self, hash_keys: list, hbase_manager: HBaseManager, index):

        if len(hash_keys) == 0:
            print("no good hash keys")
            return random.choice(self.ALL_DIGITS), self.get_default_probs()

        hash_rows = hbase_manager.batch_get_rows(self.TABLE_NAME, hash_keys)
        hashed_predictions = [hash_row.row_values[self.COLUMN_NAME] for hash_row in hash_rows]

        if len(hashed_predictions) == 0:
            print("no collision predictions")
            return random.choice(self.ALL_DIGITS), self.get_default_probs()

        test = dict((x,hashed_predictions.count(x)) for x in set(hashed_predictions))

        best_prediction = max(hashed_predictions, key=hashed_predictions.count)

        return best_prediction

    def get_hash_probabilities(self, hashed_predictions):
        prediction_count = Counter(hashed_predictions)
        prediction_probs =  {k: v / len(hashed_predictions) for k, v in prediction_count.items()}
        for digit in self.ALL_DIGITS:
            if digit not in prediction_probs:
                prediction_probs[digit] = 0.0

        return prediction_probs


    def get_default_probs(self):
        return dict([(digit, self.DEFAULT_PROB) for digit in self.ALL_DIGITS])


def main():
    node = MnistNode()
    probs = node.predict_hash_values([], None, index=1)


if __name__ == "__main__":
    main()
