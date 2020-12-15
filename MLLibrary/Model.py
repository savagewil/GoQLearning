from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, X, Y, **keyword_arguments):
        pass

    @abstractmethod
    def predict(self, X, **keyword_arguments):
        pass
