import numpy as np
from hashlearner.HashNode import HashNode
from pandas import *
import gc

class SimpleHashNode(HashNode):
    def __init__(self, predictor_indexs, response_index):
        super().__init__(predictor_indexs, response_index)
        self.predictions_map = {}
        self.response_set = ""

    def train(self, data):
        '''
        :type data_columns: DataFrame
        :type selected_predictors: array 
        :return: 
        '''

        self.response_set = list(set(data[self.response_index]))

        for row in data.iterrows():
            row = row[1]
            value = row[self.response_index]
            key = row[self.predictor_indexs].astype(str).str.cat(sep=',')
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
            return np.random.choice(self.response_set)

    def __str__(self):
        return self.predictions_map.__str__()


def main():
    hashNode = SimpleHashNode(predict_index=[0], response_index=4)
    iris_df = read_csv("iris.data.txt", header=None)
    iris_df = iris_df.sample(frac=1).reset_index(drop=True)
    iris_group = iris_df[0].unique().size
    hashNode.train(iris_df)
    for row in iris_df.iterrows():
        print(hashNode.predict(hashNode.extract_key(row)))


if __name__ == "__main__":
    main()
