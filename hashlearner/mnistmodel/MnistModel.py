from abc import ABC, abstractmethod
from array import array
from hashlearner.mnistnodes import MnistNode

class MnistModel(ABC):

    '''
    Generic Model trains and makes predictions
    '''

    RESPONSE_INDEX = 0
    PREDICTOR_INDEX = 1

    def __init__(self, data, response_indexs, predictor_indexes):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''

        self.data = data
        self.hash_node_list: array[MnistNode] = []

        super().__init__()

    @abstractmethod
    def train_model(self, data):
        pass

    @abstractmethod
    def predict_match(self, row):
        pass