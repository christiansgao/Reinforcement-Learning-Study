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
    DOWN_SCALE_RATIO = .2
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

        j = 1
        pool = Pool(10)

        for mnist_obs in mnist_data:
            number = mnist_obs[0]
            mnist_image = mnist_obs[1]
            convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.CONVOLVE_SHAPE)
            pool.apply_async(func=self.add_to_predictions, args=(convolved_images, number, j))
            # self.add_to_predictions(convolved_images, number, j)
            j += 1

        pool.close()
        pool.join()
        print("final map size:" + str(len(self.predictions_map)))

    def add_to_predictions(self, convolved_images, number, j):
        print("adding to map: " + str(j))
        for position in list(range(len(convolved_images))):
            hash_key = self.extract_key(convolved_images[position], position)
            self.predictions_map[hash_key] = number

    def extract_key(self, convolved_image, position):
        '''
        :type convolved_image: np.ndarray
        :type position:
        :return:
        '''
        image_rescaled = MnistHelper.down_scale_image(convolved_image, ratio=self.DOWN_SCALE_RATIO)
        binarize_image = MnistHelper.binarize_image(image_rescaled, threshold=self.BINARIZE_THRESHOLD)
        key = re.sub("[^0-9]+", "", str(binarize_image))
        # hash_object = hashlib.sha1(bytes(key,'utf8'))
        # hex_dig = hash_object.hexdigest()
        # hash_key = str(i) + hex_dig

        hash_key = str(position) + "-" + key
        return hash_key

    def predict(self, key):
        '''
        :type key: str
        :return: str
        '''

        if key in self.predictions_map.keys():
            return self.predictions_map.get(key)
        else:
            return None

    def predict(self, mnist_image):
        convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.CONVOLVE_SHAPE)
        images_rescaled = MnistHelper.down_scale_images(convolved_images, ratio=self.DOWN_SCALE_RATIO)
        binarize_image = MnistHelper.binarize_image(images_rescaled, threshold=self.BINARIZE_THRESHOLD)


    def make_predictions(self, mnist_data):

        for mnist_obs in mnist_data:
            number = mnist_obs[0]
            mnist_image = mnist_obs[1]
            convolved_images = MnistHelper.convolve(mnist_image, kernel_dim=self.CONVOLVE_SHAPE)

    def __str__(self):
        return self.predictions_map.__str__()

def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:100]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.2)

    # train = mnist_data
    mnist_node = SimpleMnistNode(predictor_indexs=[1], response_index=0)
    mnist_node.train_node(train)

    true_values = list(test[4])
    predictions = mnist_node.predict(test)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_values, y_pred=predictions)

    t1 = time.time()

    # print("Average Success Rate is: " + str(success_rate))
    # print("Average Random Success Rate is: " + str(rand_success_rate))
    print("Time taken: " + str(t1 - t0) + " Seconds")


if __name__ == "__main__":
    main()
