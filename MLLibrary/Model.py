from abc import ABC, abstractmethod

from MLLibrary.StatsHandler import StatsHandler


class Model(ABC):
    def __init__(self, **kwargs):
        if "statsHandler" in kwargs:
            self.statsHandler = kwargs["statsHandler"]
        else:
            self.statsHandler = StatsHandler()

    @abstractmethod
    def fit(self, X, Y, **keyword_arguments):
        pass

    @abstractmethod
    def predict(self, X, **keyword_arguments):
        pass
