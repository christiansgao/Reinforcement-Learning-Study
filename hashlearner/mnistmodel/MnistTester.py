import time
import sklearn.metrics as sk_metrics
import numpy as np

from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hashlearner.mnistmodel.HBaseMnistModel import HBaseMnistModel
from hashlearner.helper import CSVHelper

def main():
    train = MnistLoader.read_mnist(dataset="training")
    test = MnistLoader.read_mnist(dataset="testing")

    t0 = time.time()

    mnist_model = HBaseMnistModel()
    mnist_model.train_model(mnist_data=train)

    expected, test_images = MnistHelper.extract_numbers_images(test)

    print("Starting predictions")

    predictions = mnist_model.predict_from_images(test_images, use_cache=False)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)

    CSVHelper.write_predictions(expected, predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    print("true numbers: " + str(expected))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()