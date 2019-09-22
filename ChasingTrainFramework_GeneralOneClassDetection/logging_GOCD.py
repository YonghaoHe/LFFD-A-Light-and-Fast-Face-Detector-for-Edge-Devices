# -*- coding:utf-8 -*-
import logging
import os
import sys

'''
logging module
'''


def init_logging(log_file_path=None, log_file_mode='w', log_overwrite_flag=False, log_level=logging.INFO):
    # basically, the basic log offers console output
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s')
    console_handler.setFormatter(formatter)

    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(console_handler)

    if not log_file_path or log_file_path == '':
        print('No log file is specified. The log information is only displayed in console.')
        return

    # check that the log_file is already existed or not
    if not os.path.exists(log_file_path):
        location_dir = os.path.dirname(log_file_path)
        if not os.path.exists(location_dir):
            os.makedirs(location_dir)

        file_handler = logging.FileHandler(filename=log_file_path, mode=log_file_mode)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    else:
        if log_overwrite_flag:
            print('The file [%s] is existed. And it is to be handled according to the arg [file_mode](the default is \'w\').' % log_file_path)
            file_handler = logging.FileHandler(filename=log_file_path, mode=log_file_mode)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        else:
            print('The file [%s] is existed. The [overwrite_flag] is False, please change the log file name.')
            sys.exit(0)


def temp_test():
    log_file = './test.log'
    file_mode = 'w'
    init_logging(log_file_path=log_file, log_file_mode=file_mode, log_overwrite_flag=True, log_level=logging.DEBUG)


if __name__ == '__main__':
    temp_test()
    logging.info('test info')
