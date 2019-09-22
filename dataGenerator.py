import numpy as np


class ParityDataGenerator:
    def __init__(self, digit=4):
        """
        ParityDataGenerator
        generator of the required parity data with input data with the same dimensionality as self.digit
        and corresponding output label, which is 1 when input attributes have odd 1s and 0 else
        :param digit: the digit of input data
        """
        self.digit = digit

    def generate_data(self):
        pass

    def generate_attributes(self):
        pass

    def generate_labels(self):
        pass
