# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sort
   Description :
   Author :       Liangs
   date：          2019/7/28
-------------------------------------------------
   Change Activity:
                   2019/7/28:
-------------------------------------------------
"""


def sort(packed, ref, reverse=True):
    """
    Sort a series of packed list, according to a ref list.
    Also return the original index before the sort.
    """
    assert (isinstance(packed, tuple) or isinstance(packed, list)) and isinstance(ref, list)
    packed = [ref] + [range(len(ref))] + list(packed)
    sorted_packed = [list(t) for t in zip(*sorted(zip(*packed), reverse=reverse))]
    return tuple(sorted_packed[1:])


def unsort(sorted_list, oidx):
    """
    Unsort a sorted list, based on the original idx.
    """
    assert len(sorted_list) == len(oidx), "Number of list elements must match with original indices."
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted


def tensor_unsort(sorted_tensor, oidx):
    """
    Unsort a sorted tensor on its 0-th dimension, based on the original idx.
    """
    assert sorted_tensor.size(0) == len(oidx), "Number of list elements must match with original indices."
    backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    return sorted_tensor[backidx]
