from typing import Tuple

import numpy

from MLLibrary.Data import Data
from abc import ABC, abstractmethod

from MLLibrary.Models.Model import Model


class Simulation(Data, ABC):

    @abstractmethod
    def run(self, model: Model) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
        pass
