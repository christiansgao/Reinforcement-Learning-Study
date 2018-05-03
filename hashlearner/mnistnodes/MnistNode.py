from abc import ABC, abstractmethod

class MnistNode(ABC):

    DEFAULT_DECIMAL_PRECISION = 1 # precision to 1 decimal point

    '''
    Generic Mnist Node takes in predictor indexs and response indexs.
    '''

    def __init__(self, decimal_precision = DEFAULT_DECIMAL_PRECISION):
        '''
        :type hashtype: int
        :type predictor_indexs: list
        :type response_index: int
        '''
        self.decimal_precision = decimal_precision # all results get rounded to this precision
        super().__init__()

    @abstractmethod
    def train_node(self, mnist_data):
        pass

    @abstractmethod
    def predict_from_images(self, key):
        pass