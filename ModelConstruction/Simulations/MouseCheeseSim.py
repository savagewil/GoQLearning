from typing import Tuple

import numpy

from MLLibrary.Model import Model
from MLLibrary.Simulation import Simulation


class MouseCheeseSim(Simulation):
    def run(self, model: Model) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
        pass