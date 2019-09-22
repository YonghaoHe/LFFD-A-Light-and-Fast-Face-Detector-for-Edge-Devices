# coding: utf-8


class DataBatch:
    def __init__(self, mxnet_module):
        self._data = []
        self._label = []
        self.mxnet_module = mxnet_module

    def append_data(self, new_data):
        self._data.append(self.__as_ndarray(new_data))

    def append_label(self, new_label):
        self._label.append(self.__as_ndarray(new_label))

    def __as_ndarray(self, in_data):
        return self.mxnet_module.ndarray.array(in_data, self.mxnet_module.cpu())

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label
