import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model

from hashlearner.hashmodels.HashModel import HashModel
from hashlearner.hashnodes.RandomNode import RandomNode
from hashlearner.hashnodes.SimpleHashNode import SimpleHashNode
from hashlearner.helper import CSVHelper
import os

class SimpleHashModel(HashModel):
    def __init__(self, data, response_indexs, predictor_indexes=None):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''
        super().__init__(data, response_indexs, predictor_indexes)

    def smooth_model(self):
        '''
        Smooths model to give better estimates.
        :return: 
        '''

    def train_model(self):

        smooth_spread = list(range(-2,2,1))
        sd_shrink = .05
        decimal_precision = 1

        smooth_spread2 = list(range(-2,2,1))
        sd_shrink2 = .05
        decimal_precision = 1

        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[1, 2], response_index=self.response_index, smooth_spread= smooth_spread, sd_shrink=sd_shrink, decimal_precision = decimal_precision))
        self.hash_node_list.append(
             SimpleHashNode(predictor_indexs=[0, 2], response_index=self.response_index, smooth_spread= smooth_spread, sd_shrink=sd_shrink, decimal_precision = decimal_precision))
        self.hash_node_list.append(
             SimpleHashNode(predictor_indexs=[0, 3], response_index=self.response_index, smooth_spread= smooth_spread, sd_shrink=sd_shrink, decimal_precision = decimal_precision))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[1, 2], response_index=self.response_index, smooth_spread= smooth_spread, sd_shrink=sd_shrink, decimal_precision = decimal_precision))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[1, 3], response_index=self.response_index, smooth_spread= smooth_spread, sd_shrink=sd_shrink, decimal_precision = decimal_precision))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[2, 3], response_index=self.response_index, smooth_spread= smooth_spread, sd_shrink=sd_shrink, decimal_precision = decimal_precision))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[0], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[1], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[2], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[3], response_index=self.response_index))
        #self.hash_node_list.append(
        #    SimpleHashNode(predictor_indexs=[0, 1, 2, 3], response_index=self.response_index , smooth_spread= smooth_spread2, sd_shrink=sd_shrink2))

        for hash_node in self.hash_node_list:
            hash_node.train_node(data=self.data)

    def predict(self, data):
        '''
        :type data: pandas.DataFrame
        :rtype: list
        '''

        predictions = []
        for row in data.iterrows():
            candidates = {}
            # Make prediction from each model
            for hashNode in self.hash_node_list:
                extracted_row = hashNode.extract_key(row)
                prediction = hashNode.predict(hashNode.extract_key(row))
                if prediction == None:
                    continue
                if not prediction in candidates:
                    candidates[prediction] = 1
                else:
                    candidates[prediction] += 1

            prediction = max(candidates, key=candidates.get) if len(candidates) != 0 \
                else self.response_set[np.random.choice(range(len(self.response_set)))]
            predictions.append(prediction)
        return predictions

    @staticmethod
    def jackknife(iris_df):

        predictions = []

        for row_index in range(len(iris_df)):
            print("Training Hash Model: " + str(row_index))
            test = pandas.DataFrame(data=iris_df.iloc[row_index,].copy())
            test = pandas.DataFrame.transpose(test)
            train = iris_df.drop(iris_df.index[row_index])

            hash_model = SimpleHashModel(train, response_indexs=[4])
            hash_model.train_model()
            predictions.append(hash_model.predict(test)[0])

        return predictions

def main2():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
    iris_df = pandas.read_csv("/Users/christiangao/Documents/MAS/thesis/Reinforcement-Learning-Study/hashlearner/data/iris.data.txt", header=None).sample(frac=1)

    t0 = time.time()

    sk_model.train_test_split(iris_df, test_size=0.2)

    t0 = time.time()
    expected = list(iris_df[4])
    predictions = SimpleHashModel.jackknife(iris_df)

    confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)
    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    CSVHelper.write_predictions(expected, predictions, name = "iris_predictions")



    t1 = time.time()

    print("Success Rate is: " + str(success_rate))
    print("Time taken: " + str(t1 - t0) + "Seconds")

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
    iris_df = pandas.read_csv("/Users/christiangao/Documents/MAS/thesis/Reinforcement-Learning-Study/hashlearner/data/iris.data.txt", header=None).sample(frac=1)
    train, test = sk_model.train_test_split(iris_df, test_size=0.2)

    t0 = time.time()

    sk_model.train_test_split(iris_df, test_size=0.2)

    true_values = list(test[4])
    random_predictions = np.random.permutation(true_values)

    t0 = time.time()

    hash_model = SimpleHashModel(train, response_indexs=[4])
    hash_model.train_model()
    predictions = hash_model.predict(test)

    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_values, y_pred=predictions)
    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    confusion_matrix_rand = sk_metrics.confusion_matrix(y_true=true_values, y_pred=random_predictions)
    rand_classifications = np.diagonal(confusion_matrix_rand)
    rand_success_rate = sum(rand_classifications) / np.sum(confusion_matrix_rand)

    t1 = time.time()

    print("Success Rate is: " + str(success_rate))
    print("Random Success Rate is: " + str(rand_success_rate))
    print("Time taken: " + str(t1 - t0) + "Seconds")


if __name__ == "__main__":
    main2()
