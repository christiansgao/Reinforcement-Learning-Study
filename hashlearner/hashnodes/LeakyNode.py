import numpy as np
import itertools
from hashlearner.hashnodes.HashNode import HashNode
from pandas import read_csv
import gc
import os
import math


class LeakyNode(HashNode):
    '''
    Simple Hash node
    '''

    DEFAULT_LEAKY_SKIP_FREQ = 0  # skip every how much?
    DEFAULT_SMOOTH_RANGE = (-1, 1)  # How many sd on each side

    def __init__(self, predictor_indexs, response_index, decimal_precision=HashNode.DEFAULT_DECIMAL_PRECISION,
                 smooth_range=DEFAULT_SMOOTH_RANGE, skip_freq=DEFAULT_LEAKY_SKIP_FREQ):
        super().__init__(predictor_indexs, response_index, decimal_precision)
        self.predictions_map = {}
        self.response_set = ""
        self.decimal_precision = decimal_precision
        self.smooth_range = smooth_range
        self.skip_freq = skip_freq

    def train_node(self, data, deviate=True):
        '''
        :type data: DataFrame
        :type selected_predictors: array
        :rtype: None
        '''

        self.response_set = list(set(data[self.response_index]))

        self.sd_list = [np.std(column[1]) for column in data[self.predictor_indexs].iteritems()]
        increment = math.pow(.1, self.decimal_precision)
        sd_spreads = [np.arange(sd * self.smooth_range[0], sd * self.smooth_range[1], increment).tolist() for sd in self.sd_list]
        #sd_spreads = np.transpose(np.matrix(sd_spreads))

        sd_spreads = [[round((x),ndigits=self.decimal_precision) for x in spread] for spread in sd_spreads]

        for row in data.iterrows():
            row = row[1]  # type: Series
            value = row[self.response_index]
            predictors = row[self.predictor_indexs]
            key = row[self.predictor_indexs].astype(str).str.cat(sep=',')
            self.predictions_map[key] = value
            if deviate == True:
                fuzzy_predictors = [np.around(np.array(predictor) + spread, decimals=self.decimal_precision) for predictor, spread in zip(predictors,sd_spreads)]
                fuzzy_predictors = fuzzy_predictors.astype(str)
                expanded_fuzzy = list(itertools.product(*fuzzy_predictors))  # iterate through rows of fuzzy predictors
                for fuzzy_row in expanded_fuzzy:  # type: np.ndarray
                    key = ",".join(str(predictor) for predictor in fuzzy_row)
                    if key not in self.predictions_map:
                        self.predictions_map[key] = value

    gc.collect()

    def extract_key(self, row):
        row = row[1]
        key = row[self.predictor_indexs].astype(str).str.cat(sep=',')
        return key

    def predict(self, key):
        '''
        :type key: str
        :return: str
        '''

        if key in self.predictions_map.keys():
            return self.predictions_map.get(key)
        else:
            return None

    def __str__(self):
        return self.predictions_map.__str__()


def main():
    leakyNode = LeakyNode(predictor_indexs=[0], response_index=4)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root

    iris_df = read_csv(ROOT_DIR + "/data/iris.data.txt", header=None)
    iris_df = iris_df.sample(frac=1).reset_index(drop=True)
    iris_group = iris_df[0].unique().size
    leakyNode.train_node(iris_df)
    for row in iris_df.iterrows():
        print(leakyNode.predict(leakyNode.extract_key(row)))


if __name__ == "__main__":
    main()
