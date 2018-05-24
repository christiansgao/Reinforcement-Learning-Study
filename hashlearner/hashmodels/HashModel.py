from abc import ABC, abstractmethod
from hashlearner.hashnodes.HashNode import HashNode
from array import array

class HashModel(ABC):

    '''
    Generic Model trains and makes predictions
    '''

    def __init__(self, data, response_indexs, predictor_indexes):
        '''
        :type data: DataFrame
        :type predictor_indexes: list
        :type response_set: list
        :type response_indexs: list
        '''

        self.data = data
        self.response_index = response_indexs[0]
        self.hash_node_list: array[HashNode] = []
        self.response_set = list(set(data.iloc[:,self.response_index]))

        if predictor_indexes == None:
            self.predictor_indexes = list(range(0, data.shape[1]))
            self.predictor_indexes.pop(self.response_index)
        else:
            self.predictor_indexes = predictor_indexes  # type : str


        super().__init__()

    @abstractmethod
    def train_model(self, data):
        pass

    @abstractmethod
    def predict(self, row):
        pass