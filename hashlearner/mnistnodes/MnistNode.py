from abc import ABC, abstractmethod

class HashNode(ABC):

    DEFAULT_DECIMAL_PRECISION = 1 # precision to 1 decimal point

    '''
    Generic Mnist Node takes in predictor indexs and response indexs.
    '''

    def __init__(self, predictor_indexs, response_index, decimal_precision = DEFAULT_DECIMAL_PRECISION):
        '''
        :type hashtype: int
        :type predictor_indexs: list
        :type response_index: int
        '''
        self.predictor_indexs = predictor_indexs
        self.response_index = response_index
        self.decimal_precision = decimal_precision # all results get rounded to this precision


        super().__init__()

    @abstractmethod
    def train_node(self, data):
        pass

    @abstractmethod
    def extract_key(self, row):
        pass

    @abstractmethod
    def predict(self, key):
        pass