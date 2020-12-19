from typing import Tuple

import numpy

from MLLibrary.Data import Data
from abc import ABC, abstractmethod

from MLLibrary.Model import Model


class Simulation(Data, ABC):

    @abstractmethod
    def run(self, model: Model) -> Tuple[numpy.ndarray, float]:
        pass
