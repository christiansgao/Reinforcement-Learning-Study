import math
import os
import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model


class AdaBoostModel():

    def __init__(self, X_train, Y_train, X_test, Y_test):
        '''
        :type X_train: np.ndarray
        :type Y_train: np.ndarray
        '''
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.stump_h_x_predictions, self.h_x_threshold, self.h_x_sign, self.index_mapping = self._create_h_x_stumps()

        self.n = self.stump_h_x_predictions.shape[0]  # n rows
        self.p = self.stump_h_x_predictions.shape[1]  # p predictors

        self.beta = np.repeat(0., self.p).reshape((self.p, 1))  # type: np.ndarray
        self.w = np.repeat(1. / self.n, self.n).reshape((1, self.n))

        self.model_h_x_list = np.repeat(0., self.p).reshape((self.p, 1))  # type: np.ndarray
        self.model_h_x_signs = np.repeat(0., self.p).reshape((self.p, 1))  # type: np.ndarray

    def train(self, num_iterations=20):

        training_errors = []
        testing_errors = []

        iterations = list(range(0, num_iterations))

        for i in iterations:
            best_loss_index, e = self._select_best_classifier()
            self._update_model(best_loss_index, e)

            training_error = self.training_error(self.X_train, self.Y_train)
            training_errors.append(training_error)

            testing_error = self.training_error(self.X_test, self.Y_test)
            testing_errors.append(testing_error)

            print("training error iteration " + str(i) + ": " + str(training_error))
            print("testing error iteration " + str(i) + ": " + str(testing_error))

        return training_errors, testing_errors

    def partition(a):
        return {c: (a == c).nonzero()[0] for c in np.unique(a)}

    def _create_h_x_stumps(self):
        n = self.X_train.shape[0]  # n rows
        p = self.X_train.shape[1]
        stump_h_x_predictions = np.ndarray(shape=(n, 0))
        h_x_threshold = []
        h_x_sign = np.ndarray(shape=(n, 0))
        index_mapping = []
        i = 0
        for x_attr in self.X_train.T:
            attr_factors = np.unique(x_attr)
            for threshold in attr_factors:
                stump_prediction = np.array(2 * (x_attr > threshold) - 1).reshape(n, 1)
                stump_prediction2 = np.array(2 * (x_attr < threshold) - 1).reshape(n, 1)
                stump_h_x_predictions = np.append(stump_h_x_predictions, stump_prediction, axis=1)
                stump_h_x_predictions = np.append(stump_h_x_predictions, stump_prediction2, axis=1)
                h_x_threshold.append(threshold)
                h_x_threshold.append(threshold)
                h_x_sign = np.append(h_x_sign, 1)
                h_x_sign = np.append(h_x_sign, -1)
                index_mapping.append(i)
                index_mapping.append(i)

            i += 1

        return stump_h_x_predictions, h_x_threshold, h_x_sign, index_mapping
        print("Done Stumping Data.")

    def _select_best_classifier(self):
        smallest_loss = float("inf")
        error_predictions = (self.stump_h_x_predictions * self.Y_train - 1) * -.5
        losses = np.dot(self.w, error_predictions)
        best_loss_index = np.argmin(losses)
        e = losses.flatten().tolist()[best_loss_index]

        return best_loss_index, e

    def _update_model(self, m, e):
        '''
        :type m: int
        :type e: float
        :return:
        '''

        self.model_h_x_list[m] = self.h_x_threshold[m]
        self.model_h_x_signs[m] = self.h_x_sign[m]
        if e == 0: return
        beta_m = .5 * math.log((1 - e) / e)
        self.beta[m] = beta_m
        new_weights = np.ndarray(shape=(1, 0))

        for w_i, y_i, h_i_m in zip(np.nditer(self.w), np.nditer(self.Y_train),
                                   np.nditer(self.stump_h_x_predictions[:, m])):
            w = w_i * math.exp(-y_i * beta_m * h_i_m)
            new_weights = np.append(new_weights, w)

        self.w = new_weights

    def predict(self, X_test_row):
        '''
        :type X_test_row: np.ndarray
        :return:
        '''

        extracted_attr = np.array([X_test_row[index] for index in self.index_mapping]).reshape(self.p, 1)

        predictions = (2 * (extracted_attr > self.model_h_x_list) - 1) * self.model_h_x_signs
        predictions_weighted = self.beta * predictions
        prediction_final = 1 if sum(predictions_weighted) > 0 else -1

        return prediction_final

    def training_error(self, X_test, Y_test):
        '''
        :type X_test: np.array
        :type Y_test: np.array
        :rtype: float
        '''

        predictions = np.apply_along_axis(arr=X_test, func1d=self.predict, axis=1)

        error_predictions = (predictions * Y_test.T - 1) * -.5
        error_rate = np.sum(error_predictions) / len(Y_test)
        return error_rate


def main():

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
    iris_df = pandas.read_csv(ROOT_DIR+"/data/iris.data.txt", header=None).sample(frac=1)
    train, test = sk_model.train_test_split(iris_df, test_size=0.2)

    y_train = train[4]
    x_train = train[:4]

    y_test = test[4]
    x_test = test[:4]

    t0 = time.time()

    sk_model.train_test_split(iris_df, test_size=0.2)

    true_values = list(test[4])
    random_predictions = np.random.permutation(true_values)

    t0 = time.time()
    adaboost_model = AdaBoostModel(Y_train=y_train, Y_test=y_test, X_train=x_test, X_test=x_test)

    adaboost_model.train(10)
    '''predictions = adaboost_model.training_error(test)

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
    
    '''
if __name__ == "__main__":
    main()
