# -*- coding: utf-8 -*-
"""
This module provides many types of image augmentation. One can choose appropriate augmentation for
detection, segmentation and classification.
"""
import cv2
import numpy
import random


class Augmentor(object):
    """
    All augmentation operations are static methods of this class.
    """

    def __init__(self):
        pass

    @staticmethod
    def histogram_equalisation(image):
        """
        do histogram equlisation for grayscale image
        :param image: input image with single channel 8bits
        :return: processed image
        """
        if image.ndim != 2:
            print('Input image is not grayscale!')
            return None
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        result = cv2.equalizeHist(image)
        return result

    @staticmethod
    def grayscale(image):
        """
        convert BGR image to grayscale image
        :param image: input image with BGR channels
        :return:
        """
        if image.ndim != 3:
            return None
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return result

    @staticmethod
    def inversion(image):
        """
        invert the image (255-)
        :param image: input image with BGR or grayscale
        :return:
        """
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        result = 255 - image
        return result

    @staticmethod
    def binarization(image, block_size=5, C=10):
        """
        convert input image to binary image
        cv2.adaptiveThreshold is used, for detailed information, refer to opencv docs
        :param image:
        :return:
        """
        if image.ndim == 3:
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_grayscale = image

        binary_image = cv2.adaptiveThreshold(image_grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, block_size, C)
        return binary_image

    @staticmethod
    def brightness(image, min_factor=0.5, max_factor=1.5):
        '''
        adjust the image brightness
        :param image:
        :param min_factor:
        :param max_factor:
        :return:
        '''
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        factor = numpy.random.uniform(min_factor, max_factor)
        result = image * factor
        if factor > 1:
            result[result > 255] = 255
        result = result.astype(numpy.uint8)
        return result

    @staticmethod
    def saturation(image, min_factor=0.5, max_factor=1.5):
        '''
        adjust the image saturation
        :param image:
        :param min_factor:
        :param max_factor:
        :return:
        '''
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        factor = numpy.random.uniform(min_factor, max_factor)

        result = numpy.zeros(image.shape, dtype=numpy.float32)
        result[:, :, 0] = image[:, :, 0] * factor + image_gray * (1 - factor)
        result[:, :, 1] = image[:, :, 1] * factor + image_gray * (1 - factor)
        result[:, :, 2] = image[:, :, 2] * factor + image_gray * (1 - factor)
        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(numpy.uint8)
        return result

    @staticmethod
    def contrast(image, min_factor=0.5, max_factor=1.5):
        '''
        adjust the image contrast
        :param image:
        :param min_factor:
        :param max_factor:
        :return:
        '''
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_mean = numpy.mean(image_gray)
        temp = numpy.ones((image.shape[0], image.shape[1]), dtype=numpy.float32) * gray_mean
        factor = numpy.random.uniform(min_factor, max_factor)

        result = numpy.zeros(image.shape, dtype=numpy.float32)
        result[:, :, 0] = image[:, :, 0] * factor + temp * (1 - factor)
        result[:, :, 1] = image[:, :, 1] * factor + temp * (1 - factor)
        result[:, :, 2] = image[:, :, 2] * factor + temp * (1 - factor)

        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(numpy.uint8)

        return result

    @staticmethod
    def blur(image, mode='random', kernel_size=3, sigma=1):
        """

        :param image:
        :param mode: options 'normalized' 'gaussian' 'median'
        :param kernel_size:
        :param sigma: used for gaussian blur
        :return:
        """
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        if mode == 'random':
            mode = random.choice(['normalized', 'gaussian', 'median'])

        if mode == 'normalized':
            result = cv2.blur(image, (kernel_size, kernel_size))
        elif mode == 'gaussian':
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
        elif mode == 'median':
            result = cv2.medianBlur(image, kernel_size)
        else:
            print('Blur mode is not supported: %s.' % mode)
            result = image
        return result

    @staticmethod
    def rotation(image, degree=10, mode='crop', scale=1):
        """

        :param image:
        :param degree:
        :param mode: 'crop'-keep original size, 'fill'-keep full image
        :param scale:
        :return:
        """
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        h, w = image.shape[:2]
        center_x, center_y = w / 2, h / 2
        M = cv2.getRotationMatrix2D((center_x, center_y), degree, scale)

        if mode == 'crop':
            new_w, new_h = w, h
        else:
            cos = numpy.abs(M[0, 0])
            sin = numpy.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += (new_w / 2) - center_x
            M[1, 2] += (new_h / 2) - center_y

        result = cv2.warpAffine(image, M, (new_w, new_h))
        return result

    @staticmethod
    def flip(image, orientation='h'):
        '''

        :param image:
        :param orientation:
        :return:
        '''
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        if orientation == 'h':
            return cv2.flip(image, 1)
        elif orientation == 'v':
            return cv2.flip(image, 0)
        else:
            print('Unsupported orientation: %s.' % orientation)
            return image

    @staticmethod
    def resize(image, size_in_pixel=None, size_in_scale=None):
        """

        :param image:
        :param size_in_pixel: tuple (width, height)
        :param size_in_scale: tuple (width_scale, height_scale)
        :return:
        """
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        if size_in_pixel is not None:
            return cv2.resize(image, size_in_pixel)
        elif size_in_scale is not None:
            return cv2.resize(image, (0, 0), fx=size_in_scale[0], fy=size_in_scale[1])
        else:
            print('size_in_pixel and size_in_scale are both None.')
            return image

    @staticmethod
    def crop(image, x, y, width, height):
        """

        :param image:
        :param x: crop area top-left x coordinate
        :param y: crop area top-left y coordinate
        :param width: crop area width
        :param height: crop area height
        :return:
        """
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return None

        if image.ndim == 3:
            return image[y:y + height, x:x + width, :]
        else:
            return image[y:y + height, x:x + width]

    @staticmethod
    def random_crop(image, width, height):
        """

        :param image:
        :param width: crop area width
        :param height: crop area height
        :return:
        """
        if image.dtype != numpy.uint8:
            print('Input image is not uint8!')
            return False, image

        w_interval = image.shape[1] - width
        h_interval = image.shape[0] - height

        if image.ndim == 3:
            result = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        else:
            result = numpy.zeros((height, width), dtype=numpy.uint8)

        if w_interval >= 0 and h_interval >= 0:
            crop_x, crop_y = random.randint(0, w_interval), random.randint(0, h_interval)
            if image.ndim == 3:
                result = image[crop_y:crop_y + height, crop_x:crop_x + width, :]
            else:
                result = image[crop_y:crop_y + height, crop_x:crop_x + width]
        elif w_interval < 0 and h_interval >= 0:
            put_x = -w_interval / 2
            crop_y = random.randint(0, h_interval)
            if image.ndim == 3:
                result[:, put_x:put_x + image.shape[1], :] = image[crop_y:crop_y + height, :, :]
            else:
                result[:, put_x:put_x + image.shape[1]] = image[crop_y:crop_y + height, :]
        elif w_interval >= 0 and h_interval < 0:
            crop_x = random.randint(0, w_interval)
            put_y = -h_interval / 2
            if image.ndim == 3:
                result[put_y:put_y + image.shape[0], :, :] = image[:, crop_x:crop_x + width, :]
            else:
                result[put_y:put_y + image.shape[0], :] = image[:, crop_x:crop_x + width]
        else:
            put_x, put_y = -w_interval / 2, -h_interval / 2
            if image.ndim == 3:
                result[put_y:put_y + image.shape[0], put_x:put_x + image.shape[1], :] = image[:, :, :]
            else:
                result[put_y:put_y + image.shape[0], put_x:put_x + image.shape[1]] = image[:, :]

        return result

