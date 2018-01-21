from abc import ABC, abstractmethod

class HashNode(ABC):

    '''
    Generic Node takes in predictor indexs and response indexs.
    '''

    def __init__(self, predictor_indexs, response_index):
        '''
        :type hashtype: int
        :type predictor_indexs: list
        :type response_index: int
        '''
        self.hashtype = 1
        self.predictor_indexs = predictor_indexs
        self.response_index = response_index

        super().__init__()

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def extract_key(self, row):
        pass

    @abstractmethod
    def predict(self, key):
        pass