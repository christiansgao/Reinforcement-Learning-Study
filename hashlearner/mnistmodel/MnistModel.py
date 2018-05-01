from abc import ABC, abstractmethod
from array import array
from hashlearner.mnistnodes import MnistNode

class MnistModel(ABC):

    '''
    Generic Model trains and makes predictions
    '''

    RESPONSE_INDEX = 0
    PREDICTOR_INDEX = 1

    def __init__(self):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''

        self.mnist_node_list: array[MnistNode] = [] #type: list of

        super().__init__()

    @abstractmethod
    def train_model(self, data):
        pass

    @abstractmethod
    def predict_from_images(self, images):
        pass