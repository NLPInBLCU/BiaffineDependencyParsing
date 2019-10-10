import os
import time
import psutil


# ***************************************************************
class Timer(object):
    """"""

    # =============================================================
    def __init__(self, name):
        self._name = name
        self._start = None
        self._mem = None
        self.process = psutil.Process(os.getpid())

    def getmem(self):
        return self.process.memory_info().rss / (2 ** 20)

    def __enter__(self):
        self._start = time.time()
        self._mem = self.getmem()
        return self

    def __exit__(self, *args):
        seconds = time.time() - self.start
        min, sec = divmod(seconds, 60)
        hour, min = divmod(min, 60)
        print(
            '\u001b[93m{}:Time: {} hours, {} mins, {:.2f} secs ({:.2f}S); Memory: {:0.2f}Mb\u001b[0m'.format(self.name,
                                                                                                             hour, min,
                                                                                                             sec,
                                                                                                             seconds,
                                                                                                             self.getmem() - self.mem))
        return

    # =============================================================
    @property
    def name(self):
        return self._name

    @property
    def start(self):
        return self._start

    @property
    def mem(self):
        return self._mem
