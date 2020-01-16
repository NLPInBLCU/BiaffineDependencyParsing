# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logger
   Description :
   Author :       Liangs
   date：          2019/7/28
-------------------------------------------------
   Change Activity:
                   2019/7/28:
-------------------------------------------------
"""
import logging


def init_logger(logger_name, log_file=None, is_debug=False, only_console=False):
    if not only_console:
        assert log_file
    logger = logging.getLogger(logger_name)
    if is_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # handlers:
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)

    # level:
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    c_format = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    f_format = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # add
    logger.addHandler(c_handler)
    if not only_console:
        logger.addHandler(f_handler)
    logger.info('===================== NEW LOGGER =========================')
    return logger


def get_logger(logger_name):
    return logging.getLogger(logger_name)


if __name__ == '__main__':
    my_logger = init_logger('test', 'test.log')
    my_logger.info('this is a info')
