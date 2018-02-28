import numpy as np
import itertools
from hashlearner.HashNode import HashNode
from pandas import *
import gc


class SimpleHashNode(HashNode):
    DEFAILT_SD_SHRINK = .1
    DEFAULT_SMOOTH_SPREAD = list(range(-5, 5, 1))

    def __init__(self, predictor_indexs, response_index):
        super().__init__(predictor_indexs, response_index)
        self.predictions_map = {}
        self.response_set = ""
        self.smooth_spread = SimpleHashNode.DEFAULT_SMOOTH_SPREAD
        self.df_shrink = SimpleHashNode.DEFAILT_SD_SHRINK

    def train(self, data, deviate=True):
        '''
        :type data: DataFrame
        :type selected_predictors: array 
        :rtype: None
        '''

        self.response_set = list(set(data[self.response_index]))

        self.sd_list = [np.std(column[1]) for column in data[self.predictor_indexs].iteritems()]

        sd_spreads = [np.multiply(self.smooth_spread, sd * self.df_shrink) for sd in
                      self.sd_list]
        sd_spreads = np.transpose(np.asmatrix(sd_spreads))

        for row in data.iterrows():
            row = row[1] #type: Series
            value = row[self.response_index]
            predictors = row[self.predictor_indexs]
            key = row[self.predictor_indexs].astype(str).str.cat(sep=',')
            self.predictions_map[key] = value
            if deviate == True:
                fuzzy_predictors = np.matrix.round((sd_spreads + np.asmatrix(predictors)).astype(np.double), decimals=1).T #type: np.ndarray
                fuzzy_predictors = fuzzy_predictors.astype(str)
                expanded_fuzzy = list(itertools.product(*fuzzy_predictors))
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
    hashNode.train(iris_df)
    for row in iris_df.iterrows():
        print(hashNode.predict(hashNode.extract_key(row)))


if __name__ == "__main__":
    main()
