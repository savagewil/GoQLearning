from typing import  Callable, Tuple

import numpy
import math
import pygame
import Net
from formulas import sigmoid_neat, randomize, color_formula, \
    color_formula_line_helper, draw_circle, draw_line_helper, encode_list, decode_list, to_bool


class NeatLinearNet(Net.Net):
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 middle_dem: int,
                 weight_range: Tuple[float, float] = [2.0, -2.0],
                 enabled_weights: numpy.array = None,
                 activation: Callable = sigmoid_neat,
                 color_formula_param: Callable = color_formula,
                 weights: numpy.array = None):
        super(NeatLinearNet, self).__init__(in_dem, out_dem, activation, lambda x: x, color_formula_param)
        self.middle_dem: int = middle_dem

        self.score = 0
        self.input_nodes = numpy.zeros((1, self.in_dem))

        self.node_values = numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_sum = numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_back = numpy.zeros((1, self.in_dem + self.middle_dem))

        # print("LINEAR NET 1:", weights.shape, enabled_weights.shape)
        # print("LINEAR NET 2:", self.in_dem, self.middle_dem, self.out_dem)
        # print("LINEAR NET 3:", self.node_values.shape, self.node_sum.shape, self.node_back.shape)
        if weights is not None:
            self.weights = weights
        else:
            dif = abs(weight_range[0] - weight_range[1])
            self.weights = randomize(numpy.zeros((self.in_dem + self.middle_dem, self.middle_dem + self.out_dem)))
            self.weights = numpy.add(weight_range[1], numpy.multiply(dif, self.weights))

        if enabled_weights is not None:
            self.enabled_weights: numpy.array = enabled_weights
        else:
            self.enabled_weights: numpy.array = numpy.array([
                [in_node < out_node
                 for out_node in range(self.in_dem, self.in_dem + self.middle_dem + self.out_dem)]
                for in_node in range(self.in_dem + self.middle_dem)])

    def update(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, scale_dot: int = 5):

        in_spacing = (height - scale_dot * 2) / (self.in_dem + 1)
        out_spacing = (height - scale_dot * 2) / (self.out_dem + 1)

        center_x = x + width / 2  # x + width / 4
        center_y = y + height - scale_dot
        radius_y = height - scale_dot * 2
        radius_x = width / 4  # width / 2
        angle = 0
        if self.middle_dem > 1:
            angle = math.pi / (self.middle_dem - 1)

        self.in_screen_range = [screen] * self.in_dem
        self.middle_screen_range = [screen] * self.middle_dem
        self.out_screen_range = [screen] * self.out_dem
        self.line_screen_range = [[screen] * (self.out_dem + self.middle_dem)] * (self.middle_dem + self.in_dem)

        self.in_range_loc = numpy.zeros((self.in_dem, 2)).astype(int)
        self.in_range_loc[:, 0:1] = (numpy.add(x + scale_dot, self.in_range_loc[:, 0:1]))
        self.in_range_loc[:, 1:2] = (numpy.add(y + scale_dot, numpy.multiply(
            numpy.add(numpy.reshape(range(self.in_dem), (self.in_dem, 1)), 1), in_spacing)))

        self.middle_range_loc = numpy.concatenate([
            numpy.reshape(numpy.subtract(center_x, numpy.multiply(radius_x, numpy.cos(
                numpy.multiply(angle, range(self.middle_dem))))),
                          (self.middle_dem, 1)),
            numpy.reshape(numpy.subtract(center_y, numpy.multiply(radius_y, numpy.sin(
                numpy.multiply(angle, range(self.middle_dem))))),
                          (self.middle_dem, 1)),
        ], 1).astype(int)

        self.out_range_loc = numpy.zeros((self.out_dem, 2)).astype(int)
        self.out_range_loc[:, 0:1] = (numpy.add(x + width - scale_dot, self.out_range_loc[:, 0:1]))
        self.out_range_loc[:, 1:2] = (numpy.add(y + scale_dot, numpy.multiply(
            numpy.add(numpy.reshape(range(self.out_dem), (self.out_dem, 1)), 1), out_spacing)))

        self.in_radius_range = [scale_dot] * self.in_dem
        self.middle_radius_range = [scale_dot] * self.middle_dem
        self.out_radius_range = [scale_dot] * self.out_dem
        self.line_radius_range = [[1] * (self.out_dem + self.middle_dem)] * (self.middle_dem + self.in_dem)

        # self.line_start_loc = [numpy.concatenate([self.in_range_loc, self.middle_range_loc], 0)] * (
        #         self.middle_dem + self.out_dem)
        # self.line_end_loc = [[end] * (self.in_dem + self.middle_dem) for end in
        #                      numpy.concatenate([self.middle_range_loc, self.out_range_loc], 0)]

        self.line_start_loc = [[start] * (self.out_dem + self.middle_dem) for start in
                               numpy.concatenate([self.in_range_loc, self.middle_range_loc], 0)]
        self.line_end_loc = [numpy.concatenate([self.middle_range_loc, self.out_range_loc], 0)] * (
                self.middle_dem + self.in_dem)

        self.update_colors()

    def update_colors(self):
        self.in_color_range = list(map(self.color_formula, self.input_nodes[0]))
        self.middle_color_range = list(map(self.color_formula, self.node_values[0][:self.middle_dem]))
        self.out_color_range = list(map(self.color_formula, self.node_values[0][self.middle_dem:]))
        self.line_color_range = list(map(color_formula_line_helper, self.weights))

    def draw(self):
        self.update_colors()

        list(map(draw_line_helper, self.line_screen_range, self.line_color_range,
                 self.line_start_loc, self.line_end_loc, self.line_radius_range, self.enabled_weights))

        any(map(draw_circle, self.in_screen_range, self.in_color_range, self.in_range_loc, self.in_radius_range))

        any(map(draw_circle, self.middle_screen_range, self.middle_color_range, self.middle_range_loc,
                self.middle_radius_range))

        any(map(draw_circle, self.out_screen_range, self.out_color_range, self.out_range_loc,
                self.out_radius_range))

    def set_in(self, array: Tuple[float]):
        array = array
        assert len(array) == self.in_dem
        self.input_nodes = numpy.array(array, ndmin=2)
        self.node_values = numpy.zeros((1, self.middle_dem + self.out_dem))

    def get_out(self):
        # print(self.weights.shape)

        self.weights = numpy.multiply(self.weights, self.enabled_weights)
        numpy.zeros((1, self.middle_dem + self.out_dem))
        # print(self.out_dem)
        # print(self.in_dem)
        # print(self.weights.shape)
        self.node_sum = numpy.dot(self.input_nodes, self.weights[:self.in_dem])
        # print(self.node_sum)

        for i in range(self.middle_dem):
            self.node_values[0][i] = self.activation_function(self.node_sum[0][i])
            # print(self.node_values[0][i])

            self.node_sum = numpy.add(self.node_sum, numpy.multiply(self.node_values[0][i],
                                                                    self.weights[self.in_dem + i:self.in_dem + i + 1]))

        self.node_values = self.activation_function(self.node_sum)

        # print("LINEAR NET 1:", self.weights.shape, self.enabled_weights.shape)
        # print("LINEAR NET 2:", self.in_dem, self.middle_dem, self.out_dem)
        # print("LINEAR NET 3:", self.node_values.shape, self.node_sum.shape, self.node_back.shape)
        return self.node_values[0][self.middle_dem:]

    def save(self) -> str:
        weight_save = encode_list(self.weights, str, 0)
        enable_save = encode_list(self.enabled_weights, str, 0)
        save_string = "%d|%d|%d|%s|%s" % (self.in_dem - 1, self.out_dem, self.middle_dem, weight_save, enable_save)
        return save_string

    def load(self, save):
        save = save.split("|")
        in_dem = int(save[0])
        out_dem = int(save[1])
        middle_dem = int(save[2])

        weight_save = numpy.array(decode_list(save[3], float, 0))
        enable_save = decode_list(save[4], to_bool, 0)

        self.__init__(in_dem, out_dem, middle_dem, weights=weight_save, enabled_weights=enable_save)
