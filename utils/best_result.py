# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     best_result
   Description :
   Author :       Liangs
   date：          2019/7/27
-------------------------------------------------
   Change Activity:
                   2019/7/27:
-------------------------------------------------
"""


class BestResult(object):
    def __init__(self):
        self.best_LAS = -1
        self.best_UAS = -1
        self.best_LAS_step = -1
        self.best_UAS_step = -1

    def is_new_record(self, LAS, UAS, global_step):
        new_best = False
        if self.best_LAS < LAS:
            self.best_LAS = LAS
            self.best_LAS_step = global_step
            new_best = True
        if self.best_UAS < UAS:
            self.best_UAS = UAS
            self.best_UAS_step = global_step
        return new_best

    def __str__(self):
        return f'The best LAS: {self.best_LAS} in {self.best_LAS_step} (global) steps;\n' \
               f'The best UAS: {self.best_UAS} in {self.best_UAS_step} (global) steps.'


if __name__ == '__main__':
    best_result = BestResult()
    print(best_result)
