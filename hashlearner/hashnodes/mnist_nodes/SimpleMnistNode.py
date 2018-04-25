import re
import time
from multiprocessing import Pool, Manager

import numpy as np
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model

from hashlearner.hashnodes.HashNode import HashNode
from hashlearner.helper.mnist import MnistLoader, MnistHelper


class SimpleMnistNode(HashNode):
    '''
    Simple Hash node
    '''

    DEFAULT_SD_SHRINK = .1  # Amount of sd increments to smooth by
    DEFAULT_SMOOTH_SPREAD = list(range(-5, 5, 1))  # How many sd on each side
    CONVOLVE_SHAPE = (10, 10)
    DOWN_SCALE_RATIO = .5
    BINARIZE_THRESHOLD = 128

    def __init__(self, predictor_indexs, response_index):
        super().__init__(predictor_indexs, response_index)
        manager = Manager()

        self.predictions_map = manager.dict()
        self.response_set = ""

    def train_node(self, mnist_data, deviate=True):
        '''
        :type mnist_data: list of tuple
        :type deviate: boolean
        :rtype: None
        '''

        print("Starting Training for mnist data")

        j = 1
        pool = Pool(10)

        for mnist_obs in mnist_data:
            number = mnist_obs[0]
            mnist_image = mnist_obs[1]
            pool.apply_async(func=self.add_to_predictions, args=(mnist_image, number, j))
            # self.add_to_predictions(convolved_images, number, j)
            j += 1

        pool.close()
        pool.join()
        print("final map size:" + str(len(self.predictions_map)))

    def add_to_predictions(self, mnist_image: np.ndarray, number: int, j: int):
        hash_keys = self.extract_keys(mnist_image)
        pred_map = dict([(hash_key, number) for hash_key in hash_keys])
        self.predictions_map.update(pred_map)
        print("trained keys for image: " + str(j))

    def extract_keys(self, mnist_image: list):

        convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.CONVOLVE_SHAPE)
        images_rescaled = MnistHelper.down_scale_images(convolved_images, ratio=self.DOWN_SCALE_RATIO)
        binarized_images = MnistHelper.binarize_images(images_rescaled, threshold=self.BINARIZE_THRESHOLD)
        feature_positions = list(range(len(binarized_images)))

        hash_keys = [str(1) + "-" + re.sub("[^0-9]+", "", str(binarized_image)) for binarized_image in binarized_images]

        # hash_object = hashlib.sha1(bytes(key,'utf8'))
        # hex_dig = hash_object.hexdigest()
        # hash_key = str(i) + hex_dig

        return hash_keys

    def extract_key(self, row):
        pass

    def predict(self, key):
        '''
        :type key: str
        :return: str
        '''

        if key in self.predictions_map.keys():
            return self.predictions_map.get(key)
        else:
            return None

    def predict_from_image(self, mnist_image):
        print("predicting image")
        hash_keys = self.extract_keys(mnist_image)
        hashed_predictions = [self.predict(key) for key in hash_keys]
        hashed_predictions = list(filter(None.__ne__, hashed_predictions))
        best_prediction = max(hashed_predictions,key=hashed_predictions.count)

        return best_prediction

    def predict_from_images(self, mnist_images):
        pool = Pool(10)
        results = [pool.apply_async(self.predict_from_image, args=(mnist_images[0],)) for mnist_image in mnist_images]
        pool.close()
        pool.join()
        predictions = [result.get() for result in results]
        return predictions


    def __str__(self):
        return self.predictions_map.__str__()


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:20]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.2)

    #train = mnist_data
    mnist_node = SimpleMnistNode(predictor_indexs=[1], response_index=0)
    mnist_node.train_node(train)

    true_numbers = [image[0] for image in test]
    test_images = [image[1] for image in test]
    print("Starting predictions")

    predictions = mnist_node.predict_from_images(test_images)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_numbers, y_pred=predictions)

    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)

    t1 = time.time()

    print("true numbers: " + str(true_numbers))
    print("predictions: " + str(predictions))

    print("Average Success Rate is: " + str(success_rate))
    print("Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
