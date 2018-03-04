import time
import os
import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model

from hashlearner.hashmodels.HashModel import HashModel
from hashlearner.hashnodes.RandomNode import RandomNode
from hashlearner.hashnodes.LeakyNode import LeakyNode


class LeakyHashModel(HashModel):
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
        self.hash_node_list.append(
            LeakyNode(predictor_indexs=[1, 2], response_index=self.response_index))
        self.hash_node_list.append(
            LeakyNode(predictor_indexs=[0], response_index=self.response_index))
        self.hash_node_list.append(
            LeakyNode(predictor_indexs=[1], response_index=self.response_index))
        self.hash_node_list.append(
            LeakyNode(predictor_indexs=[2], response_index=self.response_index))
        self.hash_node_list.append(
            LeakyNode(predictor_indexs=[3], response_index=self.response_index))
        self.hash_node_list.append(
            RandomNode(predictor_indexs=[0, 1, 2, 3], response_index=self.response_index, hash_length=2))

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

            prediction = max(candidates, key=candidates.get) if len(candidates) != 0 else np.random.choice(
                self.response_set)
            predictions.append(prediction)
        return predictions


def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
    iris_df = pandas.read_csv(ROOT_DIR + "/data/iris.data.txt", header=None)
    train, test = sk_model.train_test_split(iris_df, test_size=0.2)

    t0 = time.time()

    sk_model.train_test_split(iris_df, test_size=0.2)

    true_values = list(test[4])
    random_predictions = np.random.permutation(true_values)

    t0 = time.time()

    hash_model = LeakyHashModel(train, response_indexs=[4])
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
    main()
