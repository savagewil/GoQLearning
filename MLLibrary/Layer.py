import types
from abc import abstractmethod
from typing import Tuple, List, Callable, Type

import numpy, random, math, pygame

from MLLibrary.Model import Model
from MLLibrary.formulas import distance_formula, sigmoid, tanh, tanh_derivative, sigmoid_der, color_formula


class Layer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def propagate_forward(self, X):
        pass

    @abstractmethod
    def propagate_backward(self, gradient):
        pass

    @abstractmethod
    def update_weights(self, learning_rate):
        pass

    @abstractmethod
    def clear(self):
        pass
