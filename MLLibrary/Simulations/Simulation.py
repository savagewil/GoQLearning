from typing import Tuple

import numpy

from MLLibrary.Data import Data
from abc import ABC, abstractmethod

from MLLibrary.Models.Model import Model


class Simulation(Data, ABC):
    def __init__(self):
        self.score = 0

    @abstractmethod
    def run(self, model: Model) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
        pass

    @abstractmethod
    def handle(self,  in_array:numpy.ndarray):
        pass

    @abstractmethod
    def state(self) -> numpy.ndarray:
        pass

    def score(self):
        return self.score
