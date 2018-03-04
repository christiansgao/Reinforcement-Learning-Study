import numpy as np
from hashlearner.hashnodes.HashNode import HashNode
from pandas import *
import random
import gc


class RandomNode(HashNode):
    def __init__(self, predictor_indexs, response_index, hash_length):
        super().__init__(predictor_indexs, response_index)
        self.predictions_map = {}
        self.response_set = ""
        self.hash_length = hash_length

    def train_node(self, data):
        '''
        :type data_columns: DataFrame
        :type selected_predictors: array 
        :return: 
        '''

        self.response_set = list(set(data[self.response_index]))

        for row in data.iterrows():
            row = row[1]
            random_indexs = np.random.choice(self.predictor_indexs, size=self.hash_length, replace=False)
            value = row[self.response_index]
            key = row[random_indexs].astype(str).str.cat(sep=',')
            self.predictions_map[key] = value
            gc.collect()

    def extract_key(self, row):
        row = row[1]
        random_indexs = np.random.choice(self.predictor_indexs, size=self.hash_length, replace=False)
        key = row[random_indexs].astype(str).str.cat(sep=',')
        return key

    def predict(self, key):
        '''
        :type key: str 
        :return: str
        '''

        if key in self.predictions_map.keys():
            return self.predictions_map.get(key)
        else:
            return np.random.choice(self.response_set)

    def __str__(self):
        return self.predictions_map.__str__()


def main():
    hashNode = RandomNode(predictor_indexs=[0, 1, 2, 3], response_index=4, hash_length=2)
    iris_df = read_csv("iris.data.txt", header=None)
    iris_df = iris_df.sample(frac=1).reset_index(drop=True)
    iris_group = iris_df[0].unique().size
    hashNode.train_node(iris_df)
    for row in iris_df.iterrows():
        print(hashNode.predict(hashNode.extract_key(row)))


if __name__ == "__main__":
    main()
