import numpy as np
import itertools
from hashlearner.hashnodes.HashNode import HashNode
from pandas import *
import gc
import math
import sklearn.model_selection as sk_model


class SimpleHashNode(HashNode):

    '''
    Simple Hash node
    '''

    DEFAULT_SD_SHRINK = .1 #Amount of sd increments to smooth by
    DEFAULT_SMOOTH_SPREAD = list(range(-5, 5, 1)) #How many sd on each side

    def __init__(self, predictor_indexs, response_index, smooth_spread = DEFAULT_SMOOTH_SPREAD, sd_shrink= DEFAULT_SD_SHRINK, decimal_precision = HashNode.DEFAULT_DECIMAL_PRECISION):
        super().__init__(predictor_indexs, response_index, decimal_precision)
        self.predictions_map = {}
        self.response_set = ""
        self.smooth_spread = smooth_spread
        self.sd_shrink = sd_shrink

    def train_node(self, data, deviate=True):
        '''
        :type data: DataFrame
        :type selected_predictors: array
        :rtype: None
        '''

        data = sk_model.train_test_split(data, test_size=0)[0]
        self.response_set = list(set(data[self.response_index]))

        self.sd_list = [np.std(column[1]) for column in data[self.predictor_indexs].iteritems()]

        sd_spreads = [np.multiply(self.smooth_spread, sd * self.sd_shrink) for sd in self.sd_list]
        sd_spreads = np.transpose(np.asmatrix(sd_spreads))

        for row in data.iterrows():
            row = row[1] #type: Series
            value = row[self.response_index]
            predictors = row[self.predictor_indexs]
            key = row[self.predictor_indexs].astype(str).str.cat(sep=',')
            self.predictions_map[key] = value
            if deviate == True:
                fuzzy_predictors = np.matrix.round((sd_spreads + np.asmatrix(predictors)).astype(np.double), decimals=self.decimal_precision).T #type: np.ndarray
                fuzzy_predictors = fuzzy_predictors.astype(str)
                expanded_fuzzy = list(itertools.product(*fuzzy_predictors)) #iterate through rows of fuzzy predictors
                for fuzzy_row in expanded_fuzzy: #type: np.ndarray
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
    hashNode = SimpleHashNode(predictor_indexs=[0], response_index=4)
    iris_df = read_csv("data/iris.data.txt", header=None)
    iris_df = iris_df.sample(frac=1).reset_index(drop=True)
    iris_group = iris_df[0].unique().size
    hashNode.train_node(iris_df)
    for row in iris_df.iterrows():
        print(hashNode.predict(hashNode.extract_key(row)))


if __name__ == "__main__":
    main()
