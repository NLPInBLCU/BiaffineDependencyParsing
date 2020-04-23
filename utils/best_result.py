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
        self.best_LAS_epoch = -1
        self.best_UAS_epoch = -1

    def is_new_record(self, LAS, UAS, epoch):
        new_best = False
        if self.best_LAS < LAS:
            self.best_LAS = LAS
            self.best_LAS_epoch = epoch
            new_best = True
        if self.best_UAS < UAS:
            self.best_UAS = UAS
            self.best_UAS_epoch = epoch
        return new_best

    def __str__(self):
        return f'The best LAS: {self.best_LAS} in {self.best_LAS_epoch} epochs;\n' \
               f'The best UAS: {self.best_UAS} in {self.best_UAS_epoch} epochs.'


if __name__ == '__main__':
    best_result = BestResult()
    print(best_result)
