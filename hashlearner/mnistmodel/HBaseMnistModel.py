import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model

from hashlearner.mnistmodel.MnistModel import MnistModel
from hashlearner.mnistnodes.MnistNode import MnistNode
from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hashlearner.mnistnodes.SimpleHBaseMnistNode import SimpleHBaseMnistNode
import os


class HBaseMnistModel(MnistModel):
    def __init__(self):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''
        super().__init__()

    def initializes_model(self):

        self.mnist_node_list.append()

    def train_model(self):

        self.initializes_model()

        for hash_node in self.mnist_node_list:
            hash_node.train_node(data=self.data)

    def predict_from_image_batch(self, images):
        '''
        :type images: pandas.DataFrame
        :rtype: list
        '''

        predictions = []
        for row in images.iterrows():
            candidates = {}
            # Make prediction from each model
            for mnistNode in self.mnist_node_list: #type: MnistNode
                extracted_row = mnistNode.extract_key(row)
                prediction = mnistNode.predict_from_images(images)
                if prediction == None:
                    raise Exception("Prediction == None!")
                if not prediction in candidates:
                    candidates[prediction] = 1
                else:
                    candidates[prediction] += 1

            prediction = max(candidates, key=candidates.get) if len(candidates) != 0 else np.random.choice(self.response_set)
            predictions.append(prediction)

        return predictions


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:100]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.1)

    mnist_node = HBaseMnistModel(predictor_indexs=[1], response_index=0)
    status = mnist_node.train_node(train)
    print("training status: " + str(status))

    true_numbers = [image[0] for image in test]
    test_images = [image[1] for image in test]
    print("Starting predictions")

    predictions = mnist_node.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_numbers, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    print("true numbers: " + str(true_numbers))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
