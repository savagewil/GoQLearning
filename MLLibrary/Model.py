from abc import ABC, abstractmethod

from MLLibrary.StatsHandler import StatsHandler


class Model(ABC):
    def __init__(self, in_size, out_size, **kwargs):
        self.in_size    = in_size
        self.out_size   = out_size
        if "statsHandler" in kwargs:
            self.statsHandler = kwargs["statsHandler"]
        else:
            self.statsHandler = StatsHandler()

    @abstractmethod
    def fit(self, data, batch=1,
            max_iterations=0, target_accuracy=None,
            batches_in_accuracy=1,
            err_der=(lambda Y, P: (Y - P) / 2.0),
            err=(lambda Y, P: (Y - P) ** 2.0), **keyword_arguments):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def predict(self, X, **keyword_arguments):
        pass
