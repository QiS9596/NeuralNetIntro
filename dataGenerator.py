import numpy as np
import pandas as pd


class ParityDataGenerator:
    def __init__(self, digit=4):
        """
        ParityDataGenerator
        generator of the required parity data with input data with the same dimensionality as self.digit
        and corresponding output label, which is 1 when input attributes have odd 1s and 0 else
        :param digit: the digit of input data
        """
        self.digit = digit
        self.max_generated_value = np.power(2, self.digit)

    def generate_data(self):
        """
        generate parity data correspond to digit
        
        :return: pandas.DataFrame object which contains the target data
        """
        attributes = self.generate_attributes()
        labels = np.array([self.generate_labels(attributes)])
        columns = ['digit%d' % (i + 1) for i in range(self.digit)]
        columns.append('label')
        data = np.concatenate([attributes, labels.T], axis=1)
        _df = pd.DataFrame(data, columns=columns)
        return _df

    def generate_attributes(self):
        """
        generate data entry
        :return:
        """
        result = []
        for i in range(self.max_generated_value):
            binary = np.binary_repr(i, self.digit)
            result.append([int(char) for char in binary])
        return np.array(result)

    def generate_labels(self, attributes):

        result = attributes.sum(axis=1, dtype=np.int32)

        return np.remainder(result, 2)

