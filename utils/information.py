# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/9
import sys
import inspect


def debug_print(message):
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    print(f'\n >>> {info.filename}, func={info.function}, line={info.lineno} :')
    print(message)


if __name__ == '__main__':
    debug_print('this')
    debug_print('is a')
    debug_print('test message')
