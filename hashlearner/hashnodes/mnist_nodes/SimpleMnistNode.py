import numpy as np
from hashlearner.hashnodes.HashNode import HashNode
import sklearn.model_selection as sk_model
from hashlearner.helper.mnist import MnistLoader, MnistHelper
import re
import time
from multiprocessing import Pool, Manager

class SimpleMnistNode(HashNode):
    '''
    Simple Hash node
    '''

    DEFAULT_SD_SHRINK = .1  # Amount of sd increments to smooth by
    DEFAULT_SMOOTH_SPREAD = list(range(-5, 5, 1))  # How many sd on each side
    CONVOLVE_SHAPE = (10, 10)

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
            #pool.apply_async(func=self.add_to_predictions, args=(convolved_images,number,j))
            self.add_to_predictions(convolved_images, number, j)
            j += 1

        pool.close()
        pool.join()
        print("final map size:" + str(len(self.predictions_map)))

    def add_to_predictions(self, convolved_images, number, j):
        print("adding to map: " + str(j))
        for i in list(range(len(convolved_images))):
            hash_key = self.extract_key(convolved_images[i], i)
            self.predictions_map[hash_key] = number

    def extract_key(self, convolved_image, i):
        '''
        :type convolved_image: np.ndarray
        :type i:
        :return:
        '''

        key = re.sub("[^0-9]+", "", str(convolved_image))
        hash_key = str(i) + str(hash(key))
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

    def __str__(self):
        return self.predictions_map.__str__()


def main():
    mnist_data = MnistLoader.read_mnist()
    mnist_data = mnist_data[:1000]

    t0 = time.time()

    train, test = sk_model.train_test_split(mnist_data, test_size=0.2)
    mnist_node = SimpleMnistNode(predictor_indexs=[1], response_index=0)
    mnist_node.train_node(train)

    t1 = time.time()


    #print("Average Success Rate is: " + str(success_rate))
    #print("Average Random Success Rate is: " + str(rand_success_rate))
    print("Time taken: " + str(t1 - t0) + " Seconds")



if __name__ == "__main__":
    main()
