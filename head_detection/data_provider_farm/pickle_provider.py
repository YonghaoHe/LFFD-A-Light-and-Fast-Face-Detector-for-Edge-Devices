'''
This provider accepts an adapter, save dataset in pickle file and load all dataset to memory for data iterators
'''

import cv2
import numpy
import pickle

from ChasingTrainFramework_GeneralOneClassDetection.data_provider_base.base_provider import ProviderBaseclass
from .text_list_adapter import TextListAdapter


class PickleProvider(ProviderBaseclass):
    """
    This class provides methods to save and read data.
    By default, images are compressed using JPG format.
    If data_adapter is not None, it means saving data, or it is reading data
    """

    def __init__(self,
                 pickle_file_path,
                 encode_quality=90,
                 data_adapter=None):
        ProviderBaseclass.__init__(self)

        if data_adapter:  # write data

            self.data_adapter = data_adapter
            self.data = {}
            self.counter = 0
            self.pickle_file_path = pickle_file_path

        else:  # read data

            self.data = pickle.load(open(pickle_file_path, 'rb'))
            # get positive and negative indeices
            self._positive_index = []
            self._negative_index = []
            for k, v in self.data.items():
                if v[1] == 0:  # negative
                    self._negative_index.append(k)
                else:  # positive
                    self._positive_index.append(k)

        self.compression_mode = '.jpg'
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, encode_quality]

    @property
    def positive_index(self):
        return self._positive_index

    @property
    def negative_index(self):
        return self._negative_index

    def write(self):

        for data_item in self.data_adapter.get_one():

            temp_sample = []
            im, bboxes = data_item
            ret, buf = cv2.imencode(self.compression_mode, im, self.encode_params)
            if buf is None or buf.size == 0:
                print('buf is wrong.')
                continue
            if not ret:
                print('An error is occurred.')
                continue
            temp_sample.append(buf)

            if isinstance(bboxes, str):  # 负样本
                temp_sample.append(0)
                temp_sample.append(int(bboxes))
            else:
                temp_sample.append(1)
                temp_sample.append(bboxes)

            self.data[self.counter] = temp_sample
            print('Successfully save the %d-th data item.' % self.counter)
            self.counter += 1

        pickle.dump(self.data, open(self.pickle_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def read_by_index(self, index):
        im_buf, flag, bboxes = self.data[index]
        im = cv2.imdecode(im_buf, cv2.IMREAD_COLOR)
        return im, flag, bboxes


def write_file():
    data_list_file_path = './data_folder/data_list_brainwash_test.txt'
    adapter = TextListAdapter(data_list_file_path)

    pickle_file_path = './data_folder/data_list_brainwash_test.pkl'
    encode_quality = 90
    packer = PickleProvider(pickle_file_path, encode_quality, adapter)
    packer.write()


def read_file():
    pickle_file_path = './data_folder/data_list_brainwash_test.pkl'

    provider = PickleProvider(pickle_file_path)
    positive_index = provider.positive_index
    negative_index = provider.negative_index
    print("num of positive: %d\nnum of negative: %d" % (len(positive_index), len(negative_index)))
    # all_index = positive_index+negative_index
    import random
    random.shuffle(positive_index)

    for i, index in enumerate(positive_index):
        im, flag, bboxes_numpy = provider.read_by_index(index)
        if isinstance(bboxes_numpy, numpy.ndarray):
            for n in range(bboxes_numpy.shape[0]):
                cv2.rectangle(im, (bboxes_numpy[n, 0], bboxes_numpy[n, 1]),
                              (bboxes_numpy[n, 0] + bboxes_numpy[n, 2], bboxes_numpy[n, 1] + bboxes_numpy[n, 3]), (0, 255, 0), 1)
        cv2.imshow('im', im)
        cv2.waitKey()


if __name__ == '__main__':
    # write_file()
    read_file()
