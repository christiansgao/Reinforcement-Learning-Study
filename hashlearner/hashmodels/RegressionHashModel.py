import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model

from hashlearner.hashmodels.HashModel import HashModel
from hashlearner.hashnodes.RandomNode import RandomNode
from hashlearner.hashnodes.SimpleHashNode import SimpleHashNode
import os
import matplotlib.pyplot as pyplot
import math
import sklearn.linear_model as linear_model


class RegressionHashModel(HashModel):
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

        smooth_spread = list(range(-1, 1, 1))
        sd_shrink = .001
        decimal_precision = 1

        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[0], response_index=self.response_index, smooth_spread=smooth_spread,
                           sd_shrink=sd_shrink, decimal_precision=decimal_precision))

        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[0], response_index=self.response_index, smooth_spread=smooth_spread,
                           sd_shrink=sd_shrink, decimal_precision=decimal_precision))

        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[0], response_index=self.response_index, smooth_spread=smooth_spread,
                           sd_shrink=sd_shrink, decimal_precision=decimal_precision))

        self.hash_node_list.append(
            SimpleHashNode(predictor_indexs=[0], response_index=self.response_index, smooth_spread=smooth_spread,
                           sd_shrink=sd_shrink, decimal_precision=decimal_precision))

        for hash_node in self.hash_node_list:
            hash_node.train_node(data=self.data)

    def predict(self, data):
        '''
        :type data: pandas.DataFrame
        :rtype: list
        '''

        self.response_set = list(set(data[self.response_index]))


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

            if len(candidates) != 0:
                values = [a*b for a,b in zip(list(candidates.values()),list(candidates.keys()))]
                prediction = np.mean(values)
            else: prediction = np.random.choice(self.response_set)
            #prediction = max(candidates, key=candidates.get) if len(candidates) != 0 else np.random.choice(self.response_set)
            predictions.append(prediction)
        return predictions


def show_plot(x: list, y: list, title):
    pyplot.plot(x, y, 'ro')
    pyplot.xlim(4,8)
    pyplot.ylim(1.5, 4.6)
    pyplot.title(title)
    pyplot.show()


def plot_node(node: SimpleHashNode):
    x = node.predictions_map.keys()
    y = node.predictions_map.values()
    show_plot(x,y,"Hashnode Hashmap Values")


def sum_sqr(predictions, true_values):
    return sum([math.pow(prediction - true_value, 2) for prediction, true_value in zip(predictions, true_values)])

def linear_regression(x, y, test_x, test_y):
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error

    # Make predictions using the testing set
    reg_y_pred = regr.predict(test_x.reshape(-1, 1))

    return reg_y_pred

def run_test(iris_df):
    train, test = sk_model.train_test_split(iris_df, test_size=0.2)

    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]

    true_values = list(test[1])
    random_predictions = np.random.permutation(true_values)

    hash_model = RegressionHashModel(train, response_indexs=[1], predictor_indexes=[0])
    hash_model.train_model()
    hash_pred = hash_model.predict(test)

    show_plot(iris_df[0], iris_df[1], "All data")

    plot_node(hash_model.hash_node_list[0])

    sum_squares_hash = sum_sqr(hash_pred, true_values)

    show_plot(test_x, hash_pred, "Hashnode Predictions")


    reg_y_pred = linear_regression(train_x,train_y,test_x=test_x,test_y=test_y)
    show_plot(test_x, reg_y_pred, "Regression Predictions")

    sum_squares_regession = sum_sqr(test_y, reg_y_pred)

    return sum_squares_hash, sum_squares_regession

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
    iris_df = pandas.read_csv(ROOT_DIR + "/data/iris.data.txt", header=None).sample(frac=1)
    iterations = 1
    hash_errors = []
    reg_errors = []

    for i in range(0,iterations):
        print("iteration: " + str(i))
        sum_squares_hash, sum_squares_regession = run_test(iris_df)
        hash_errors.append(sum_squares_hash)
        reg_errors.append(sum_squares_regession)

    average_hash_error = sum(hash_errors)/iterations
    average_reg_error = sum(reg_errors)/iterations


    print("Average HashNode Sum Squared Errors is: " + str(average_hash_error))
    print("Average Linear Regression Sum Squared Errors is: " + str(average_reg_error))


if __name__ == "__main__":
    main()
