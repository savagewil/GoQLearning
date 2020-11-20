from abc import abstractmethod
from typing import Tuple, List, Callable, Type

import numpy, random, math, pygame

from MLLibrary.formulas import distance_formula, sigmoid, sigmoid_der, color_formula


class Net:
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula):
        self.in_dem: int = in_dem
        self.out_dem: int = out_dem
        self.score: float = 0
        self.activation_function: Callable = activation
        self.activation_derivative: Callable = activation_der
        self.color_formula: Callable = color_formula_param

    @abstractmethod
    def set_in(self, array: List[float]):
        pass

    @abstractmethod
    def get_out(self):
        pass

    @abstractmethod
    def learn(self, ratio: float, target: List[int]):
        pass

    @abstractmethod
    def save(self) -> str:
        pass

    @abstractmethod
    def load(self, save):
        pass

    @abstractmethod
    def update(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, scale_dot: int = 5):
        pass

    @abstractmethod
    def update_colors(self):
        pass

    @abstractmethod
    def draw(self):
        pass

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __le__(self, other):
        return self.score <= other.score

    def __add__(self, other):
        if isinstance(other, Net):
            return self.score + other.score
        else:
            return self.score + other

    def __radd__(self, other):
        if isinstance(other, Net):
            return self.score + other.score
        else:
            return self.score + other
