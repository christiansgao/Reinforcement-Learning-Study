import time
from multiprocessing.pool import Pool

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model
import hashlearner.helper.RunAsynchTests as RunAsynchTests
from hashlearner.hashmodels.SimpleHashModel import SimpleHashModel


def run_model(df, i):
    np.random.seed()
    train, test = sk_model.train_test_split(df, test_size=0.2)

    true_values = list(test[4])
    random_predictions = np.random.permutation(true_values)

    hash_model = SimpleHashModel(train, response_indexs=[4])
    hash_model.train_model()
    predictions = hash_model.predict(test)

    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_values, y_pred=predictions)
    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    confusion_matrix_rand = sk_metrics.confusion_matrix(y_true=true_values, y_pred=random_predictions)
    rand_classifications = np.diagonal(confusion_matrix_rand)
    rand_success_rate = sum(rand_classifications) / np.sum(confusion_matrix_rand)
    print("Iteration: " + str(i))

    return success_rate, rand_success_rate


def main():
    iris_df = pandas.read_csv("data/iris.data.txt", header=None).sample(frac=1)
    test_iterations = 100
    t0 = time.time()

    trained, random = RunAsynchTests.run_asynch_test(function=run_model, dataset=iris_df, iterations=test_iterations)

    t1 = time.time()

    success_rate = sum(trained) / test_iterations
    rand_success_rate = sum(random) / test_iterations

    print("Average Success Rate is: " + str(success_rate))
    print("Average Random Success Rate is: " + str(rand_success_rate))
    print("Time taken: " + str(t1 - t0) + "Seconds")


if __name__ == "__main__":
    main()
