"""
This module is to read, modify and return a single sample.
It only works in data packing phase.
"""


class DataAdapterBaseclass(object):

    def __init__(self):
        pass

    def __del__(self):
        pass

    def get_one(self):
        """
        return only one sample each time
        :return:
        """
        raise NotImplementedError()
