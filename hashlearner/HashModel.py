import pandas
from array import array
from hashlearner.HashNode import HashNode
from hashlearner.SimpleHashNode import SimpleHashNode
import sklearn.metrics as sk_metrics
import numpy as np
import sklearn.model_selection as sk_model
from hashlearner.RandomNode import RandomNode
import time


class HashModel:
    def __init__(self, data, response_indexs, predictor_indexes=None):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''

        self.data = data
        self.response_index = response_indexs[0]
        self.hash_node_list: array[HashNode] = []

        if predictor_indexes is None:
            self.predictor_indexes = list(range(0, data.shape[1]))
            self.predictor_indexes.pop(self.response_index)
        else:
            self.predictor_indexes = predictor_indexes  # type : str

    def smooth_model(self):
        '''
        Smooths model to give better estimates.
        :return: 
        '''

    def train_model(self):
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[1, 2], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[0], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[1], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[2], response_index=self.response_index))
        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[3], response_index=self.response_index))
        self.hash_node_list.append(
            RandomNode(predictor_indexs=[0, 1, 2, 3], response_index=self.response_index, hash_length=2))

        for hash_node in self.hash_node_list:
            hash_node.train(data=self.data)

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

            prediction = max(candidates, key=candidates.get) if len(candidates) != 0 else np.random.choice(self.response_set)
            predictions.append(prediction)
        return predictions


def main():
    iris_df = pandas.read_csv("iris.data.txt", header=None).sample(frac=1)
    train, test = sk_model.train_test_split(iris_df, test_size=0.2)

    t0 = time.time()

    sk_model.train_test_split(iris_df, test_size=0.2)

    true_values = list(test[4])
    random_predictions = np.random.permutation(true_values)

    t0 = time.time()

    hash_model = HashModel(train, response_indexs=[4])
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
