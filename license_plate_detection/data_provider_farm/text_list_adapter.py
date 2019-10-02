'''
This adapter accepts a text as input which describes the annotated data.
Each line in text are formatted as:
[image absolute path],[pos/neg flag],[num of bboxes],[x1],[y1],[width1],[height1],[x2],[y2],[width2],[height2]......
'''

import cv2
import numpy
from ChasingTrainFramework_GeneralOneClassDetection.data_provider_base.base_data_adapter import DataAdapterBaseclass
import random


class TextListAdapter(DataAdapterBaseclass):

    def __init__(self, data_list_file_path):

        DataAdapterBaseclass.__init__(self)
        fin = open(data_list_file_path, 'r')
        self.lines = fin.readlines()
        fin.close()
        self.line_counter = 0

    def __del__(self):
        pass

    def get_one(self):
        """
        This function use 'yield' to return samples
        """
        while self.line_counter < len(self.lines):

            line = self.lines[self.line_counter].strip('\n').split(',')
            if line[1] == '1':  # 如果是正样本，需要校验bbox的个数是否一样
                assert len(line[3:]) == 4 * int(line[2])

            im = cv2.imread(line[0], cv2.IMREAD_UNCHANGED)

            if line[1] == '0':
                yield im, '0'
                self.line_counter += 1
                continue

            num_bboxes = int(line[2])
            bboxes = []
            for i in range(num_bboxes):
                x = float(line[3 + i * 4])
                y = float(line[3 + i * 4 + 1])
                width = float(line[3 + i * 4 + 2])
                height = float(line[3 + i * 4 + 3])

                bboxes.append([x, y, width, height])

            bboxes = numpy.array(bboxes, dtype=numpy.float32)
            yield im, bboxes

            # generate negative samples
            left = numpy.min(bboxes[:, 0])
            top = numpy.min(bboxes[:, 1])
            right = numpy.max(bboxes[:, 0] + bboxes[:, 2])
            bottom = numpy.max(bboxes[:, 1] + bboxes[:, 3])
            if random.random() < 0.25:
                im_crop = im[:, :int(left), :].copy()
                if im_crop.shape[1] > 100:
                    yield im_crop, '0'
            if random.random() < 0.25:
                im_crop = im[:, int(right):, :].copy()
                if im_crop.shape[1] > 100:
                    yield im_crop, '0'
            if random.random() < 0.25:
                im_crop = im[:int(top), :, :].copy()
                if im_crop.shape[0] > 100:
                    yield im_crop, '0'
            if random.random() < 0.25:
                im_crop = im[int(bottom):, :, :].copy()
                if im_crop.shape[0] > 100:
                    yield im_crop, '0'

            self.line_counter += 1


if __name__ == '__main__':
    pass
