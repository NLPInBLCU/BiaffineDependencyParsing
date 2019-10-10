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


def init_logger(logger_name, log_file, is_debug=False):
    logger = logging.getLogger(logger_name)
    if is_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # handlers:
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)

    # level:
    c_handler.setLevel(logging.WARNING)

    c_format = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    f_format = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # add
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


def get_logger(logger_name):
    return logging.getLogger(logger_name)
