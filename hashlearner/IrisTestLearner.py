import pandas
from hashlearner.HashModel import HashModel
from array import array
from hashlearner.HashNode import HashNode
from hashlearner.SimpleHashNode import SimpleHashNode
import sklearn.metrics as sk_metrics
import numpy as np
import sklearn.model_selection as sk_model
from hashlearner.RandomNode import RandomNode
import time
import _thread

def test_model(df, i):
    '''
    :type df: pandas.DataFrame
    :return: tuple
    '''

    try:

        train, test = sk_model.train_test_split(df, test_size=0.2)

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

        print("Iteration: " + str(1))

        return success_rate, rand_success_rate

    except Exception:
        import traceback
        print(traceback.format_exc())

    return success_rate, rand_success_rate

def main():

    iris_df = pandas.read_csv("iris.data.txt", header=None).sample(frac=1)

    test_iterations = 100
    t0 = time.time()

    trained = []
    random = []

    #for i in range(test_iterations):

    train, rand = _thread.start_new_thread(test_model, (iris_df, 1))
    trained

    t1 = time.time()

    success_rate = sum(trained)/test_iterations
    rand_success_rate = sum(random)/test_iterations

    print("Success Rate is: " + str(success_rate))
    print("Random Success Rate is: " + str(rand_success_rate))
    print("Time taken: " + str(t1 - t0) + "Seconds")

if __name__ == "__main__":
    main()
