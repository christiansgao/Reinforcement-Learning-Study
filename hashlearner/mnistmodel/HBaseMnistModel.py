import time

import numpy as np
import pandas
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model
import array
import itertools


from hashlearner.mnistmodel.MnistModel import MnistModel
from hashlearner.mnistnodes.MnistNode import MnistNode
from hashlearner.helper.mnist import MnistLoader, MnistHelper
from hashlearner.mnistnodes.SimpleHBaseMnistNode import SimpleHBaseMnistNode
from hashlearner.helper import CSVHelper
from hashlearner.mnistnodes.FilterHBaseMnistNode import FilterHBaseMnistNode


class HBaseMnistModel(MnistModel):
    def __init__(self):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''
        super().__init__()

        self.initializes_model()

    def initializes_model(self):

        #self.mnist_node_list.append(SimpleHBaseMnistNode()) #Model 1
        #self.mnist_node_list.append(SimpleHBaseMnistNode(convolve_shape=(5, 9), binarize_threshold=160, down_scale_ratio=.6))
        #self.mnist_node_list.append(SimpleHBaseMnistNode(convolve_shape=(8, 13), binarize_threshold=200, down_scale_ratio=.4, setup_table=True, table_name="simple-node-1"))
        #self.mnist_node_list.append(SimpleHBaseMnistNode(convolve_shape=(4, 6), binarize_threshold=200, down_scale_ratio=.8, setup_table=True, table_name="simple-node-2"))
        #self.mnist_node_list.append(SimpleHBaseMnistNode(convolve_shape=(12, 6), binarize_threshold=150, down_scale_ratio=.5))
        self.mnist_node_list.append(FilterHBaseMnistNode(setup_table=True))

    def train_model(self, mnist_data):

        #SimpleHBaseMnistNode().setup()

        index = 1
        for hash_node in self.mnist_node_list: #type: MnistNode
            print("####### Training Node Number: {} #######".format(str(index)))
            hash_node.train_node(mnist_data)
            index += 1

    def predict_from_images(self, images):
        '''
        :type images: pandas.DataFrame
        :rtype: list
        '''

        final_predictions = []
        candidates = [{} for _ in range(len(images))]

        for mnistNode in self.mnist_node_list: #type: SimpleHBaseMnistNode
            #print("########## Extracting Predictions for Node: {} ##########".format(mnistNode.node_name))
            predictions = mnistNode.predict_from_images(images)

            for prediction, candidate in zip(predictions, candidates):
                if not prediction in candidate:
                    candidate[prediction] = 1
                else:
                    candidate[prediction] += 1

        for candidate in candidates:
            final_prediction = max(candidate, key=candidate.get) if len(candidates) != 0 else np.random.choice(self.response_set)
            final_predictions.append(final_prediction)


        print("final candidates: " + str(candidates))

        return final_predictions


def main():

    mnist_data = MnistLoader.read_mnist()

    t0 = time.time()

    #train, test = sk_model.train_test_split(mnist_data, test_size=0.1)
    #train = mnist_data[:100]
    #test = mnist_data[200:300]
    train = MnistLoader.read_mnist(dataset="training")
    test = MnistLoader.read_mnist(dataset="testing")

    mnist_model = HBaseMnistModel()
    status = mnist_model.train_model(train)

    expected, test_images = MnistHelper.extract_numbers_images(test)

    print("Starting predictions")

    predictions = mnist_model.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    CSVHelper.write_predictions(expected, predictions, name="F-10-10_80_5")

    print("true numbers: " + str(expected))
    print("predictions: " + str(predictions))


    print("Average Success Rate is: " + str(success_rate))

    t1 = time.time()
    print("Total Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
