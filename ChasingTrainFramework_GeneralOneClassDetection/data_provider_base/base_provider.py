"""
This module takes an adapter as data supplier, pack data and provide data for data iterators

"""


class ProviderBaseclass(object):
    """
    This is the baseclass of packer. Any other detailed packer must inherit this class.
    """

    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __del__(self):
        pass

    def write(self):
        """
        Write a single sample to the files
        :return:
        """
        raise NotImplementedError()

    def read_by_index(self, index):
        """
        Read a single sample
        :return:
        """
        raise NotImplementedError()


if __name__ == '__main__':
    provider = ProviderBaseclass()
    print(provider)
