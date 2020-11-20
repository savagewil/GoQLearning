from abc import abstractmethod
from typing import Tuple, List, Callable, Type

import numpy, random, math, pygame
from MLLibrary.Net import Net
from MLLibrary.formulas import distance_formula, sigmoid, tanh, tanh_derivative, sigmoid_der, color_formula


class LSTMNet(Net):
    def __init__(self, in_dem: int,
                 out_dem: int,
                 learning_distance: int = 0,
                 color_formula_param: Callable = color_formula,
                 table_block: int = 20):
        super().__init__(in_dem, out_dem, tanh, tanh_derivative, color_formula_param)
        self.in_dem: int        = in_dem
        self.out_dem: int       = out_dem
        self.joint_dem: int     = in_dem+out_dem

        self.weights_forget     = numpy.random.random((self.joint_dem, self.out_dem))
        self.weights_store      = numpy.random.random((self.joint_dem, self.out_dem))
        self.weights_cell       = numpy.random.random((self.joint_dem, self.out_dem))
        self.weights_output     = numpy.random.random((self.joint_dem, self.out_dem))

        self.learning_distance = learning_distance
        self.table_block = table_block
        self.index = 0

        if learning_distance > 0:
            self.forget_gate    = numpy.zeros((learning_distance, self.out_dem))
            self.store_gate     = numpy.zeros((learning_distance, self.out_dem))
            self.output_gate    = numpy.zeros((learning_distance, self.out_dem))

            self.input_state    = numpy.zeros((learning_distance, self.in_dem))
            self.joint_state    = numpy.zeros((learning_distance, self.joint_dem))

            self.cell_state     = numpy.zeros((learning_distance, self.out_dem))
            self.cell_temp      = numpy.zeros((learning_distance, self.out_dem))
            self.output_state   = numpy.zeros((learning_distance, self.out_dem))
            self.mem_size = learning_distance
        else:
            self.forget_gate    = numpy.zeros((table_block, self.out_dem))
            self.store_gate     = numpy.zeros((table_block, self.out_dem))
            self.output_gate    = numpy.zeros((table_block, self.out_dem))

            self.input_state    = numpy.zeros((table_block, self.in_dem))
            self.joint_state    = numpy.zeros((table_block, self.joint_dem))

            self.cell_state     = numpy.zeros((table_block, self.out_dem))
            self.cell_temp      = numpy.zeros((table_block, self.out_dem))
            self.output_state   = numpy.zeros((table_block, self.out_dem))
            self.mem_size = table_block

        self.score: float = 0

        self.color_formula: Callable = color_formula_param

    def set_in(self, array: List[float]):
        if (1, self.in_dem) == numpy.shape(array):
            self.add_new_memory()
            index_mod = self.index % self.mem_size
            self.input_state[index_mod, :] = array
        else:
            raise ValueError("Array must be (1,%d) it is %s" % (self.in_dem, str(numpy.shape(array))))

    def add_new_memory(self):
        if self.learning_distance > 0:
            self.index += 1
            new_mem = self.index % self.mem_size
            self.forget_gate[new_mem, :] = 0
            self.store_gate[new_mem, :] = 0
            self.output_gate[new_mem, :] = 0

            self.input_state[new_mem, :] = 0
            self.joint_state[new_mem, :] = 0

            self.cell_state[new_mem, :] = 0
            self.cell_temp[new_mem, :] = 0
            self.output_state[new_mem, :] = 0
        else:
            self.index += 1
            if self.index == self.mem_size:
                forget_gate_temp = self.forget_gate
                store_gate_temp = self.store_gate
                output_gate_temp = self.output_gate

                input_state_temp = self.input_state
                joint_state_temp = self.joint_state

                cell_state_temp = self.cell_state
                cell_temp_temp = self.cell_temp
                output_state_temp = self.output_state

                old_mem_size = self.mem_size
                self.mem_size = self.mem_size + self.table_block

                self.forget_gate = numpy.zeros((self.mem_size, self.out_dem))
                self.store_gate = numpy.zeros((self.mem_size, self.out_dem))
                self.output_gate = numpy.zeros((self.mem_size, self.out_dem))

                self.input_state = numpy.zeros((self.mem_size, self.in_dem))
                self.joint_state = numpy.zeros((self.mem_size, self.joint_dem))

                self.cell_state = numpy.zeros((self.mem_size, self.out_dem))
                self.cell_temp = numpy.zeros((self.mem_size, self.out_dem))
                self.output_state = numpy.zeros((self.mem_size, self.out_dem))

                # Move old data into memory

                self.forget_gate[old_mem_size, :] = forget_gate_temp
                self.store_gate[old_mem_size, :] = store_gate_temp
                self.output_gate[old_mem_size, :] = output_gate_temp

                self.input_state[old_mem_size, :] = input_state_temp
                self.joint_state[old_mem_size, :] = joint_state_temp

                self.cell_state[old_mem_size, :] = cell_state_temp
                self.cell_temp[old_mem_size, :] = cell_temp_temp
                self.output_state[old_mem_size, :] = output_state_temp

    def get_out(self):
        index_mod = self.index % self.mem_size
        index_before = (self.index - 1) % self.mem_size

        self.joint_state[index_mod, :self.in_dem] = self.input_state[index_mod, :]
        self.joint_state[index_mod, self.in_dem:] = self.output_state[index_before, :]

        self.forget_gate[index_mod, :] = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                           self.weights_forget))
        self.store_gate[index_mod, :] = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                          self.weights_store))
        self.output_gate[index_mod, :] = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                           self.weights_output))
        self.cell_temp[index_mod, :] = tanh(numpy.dot(self.joint_state[index_mod, :],
                                                      self.weights_cell))
        self.cell_state[index_mod, :] = numpy.add(numpy.multiply(self.cell_state[index_before, :],
                                                                 self.forget_gate[index_mod, :]),
                                                  numpy.multiply(self.cell_temp[index_mod, :],
                                                                 self.store_gate[index_mod, :]))

        self.output_state[index_mod, :] = numpy.multiply(self.cell_state[index_mod, :],
                                                         self.output_gate[index_mod, :])

        return self.output_state[index_mod, :]

    def learn(self, ratio: float, derivative: List[int]):
        learning_distance = self.learning_distance
        if learning_distance <= 0:
            learning_distance = self.index
        else:
            learning_distance = min(self.learning_distance, self.index)
        d_joint_state       = numpy.zeros((learning_distance + 1, self.joint_dem))
        d_output_state      = numpy.zeros((learning_distance + 1, self.out_dem))
        d_input_state       = numpy.zeros((learning_distance + 1, self.in_dem))
        d_cell_state        = numpy.zeros((learning_distance + 1, self.out_dem))

        d_weights_forget    = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        d_weights_output    = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        d_weights_store     = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        d_weights_cell      = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        d_output_state[learning_distance] = derivative
        for index in range(1, learning_distance):
            d_index = learning_distance - index
            s_index = (self.index - 1 - index) % self.mem_size
            s_index_less = (self.index - 2 - index) % self.mem_size

            d_temp = self.output_gate[s_index] * d_output_state[d_index + 1] + d_cell_state[d_index + 1]
            d_joint_o = numpy.dot(
                d_output_state[d_index + 1] * self.output_gate[s_index] * (1.0 - self.output_gate[s_index]) * self.cell_state[s_index],
                numpy.transpose(self.weights_output)
            )
            d_joint_s = numpy.dot(
                d_temp * self.store_gate[s_index] * (1.0 - self.store_gate[s_index]) * self.cell_temp[s_index],
                numpy.transpose(self.weights_store)
            )
            d_joint_f = numpy.dot(
                d_temp * self.forget_gate[s_index] * (1.0 - self.forget_gate[s_index]) * self.cell_state[s_index_less],
                numpy.transpose(self.weights_forget)
            )
            d_joint_c = numpy.dot(
                d_temp * self.store_gate[s_index] * (1.0 - (self.cell_temp[s_index] * self.cell_temp[s_index])),
                numpy.transpose(self.weights_cell)
            )

            d_joint_state[d_index]    = d_joint_o + d_joint_s + d_joint_f + d_joint_c
            d_output_state[d_index]   = d_joint_state[d_index, self.in_dem:]
            d_input_state[d_index]    = d_joint_state[d_index, :self.in_dem]
            d_cell_state[d_index]     = self.forget_gate[s_index] * d_temp

            d_weights_output[d_index] = numpy.dot(
                numpy.transpose(self.joint_state),
                d_output_state[d_index + 1] * self.cell_state[s_index] * self.output_gate[s_index] * (1.0 - self.output_gate[s_index]))
            d_weights_forget[d_index] = numpy.dot(
                numpy.transpose(self.joint_state),
                d_temp * self.cell_state[s_index_less] * self.forget_gate[s_index] * (1.0 - self.forget_gate[s_index]))
            d_weights_store[d_index]  = numpy.dot(
                numpy.transpose(self.joint_state),
                d_temp * self.cell_temp[s_index] * self.store_gate[s_index] * (1.0 - self.store_gate[s_index]))
            d_weights_cell[d_index]   = numpy.dot(
                numpy.transpose(self.joint_state),
                d_temp * self.store_gate[s_index] * (1.0 - (self.cell_temp[s_index] * self.cell_temp[s_index])))

        # Last Row needs zeros
        d_index = 0
        s_index = (self.index - 1 - learning_distance) % self.mem_size

        d_temp = self.output_gate[s_index] * d_output_state[d_index + 1] + d_cell_state[d_index + 1]
        d_joint_o = numpy.dot(
            d_output_state[d_index + 1] * self.output_gate[s_index] * (1.0 - self.output_gate[s_index]) *
            self.cell_state[s_index],
            numpy.transpose(self.weights_output)
        )
        d_joint_s = numpy.dot(
            d_temp * self.store_gate[s_index] * (1.0 - self.store_gate[s_index]) * self.cell_temp[s_index],
            numpy.transpose(self.weights_store)
        )
        d_joint_c = numpy.dot(
            d_temp * self.store_gate[s_index] * (1.0 - (self.cell_temp[s_index] * self.cell_temp[s_index])),
            numpy.transpose(self.weights_cell)
        )

        d_joint_state[d_index] = d_joint_o + d_joint_s + d_joint_c
        d_output_state[d_index] = d_joint_state[d_index, self.in_dem:]
        d_input_state[d_index] = d_joint_state[d_index, :self.in_dem]
        d_cell_state[d_index] = self.forget_gate[s_index] * d_temp

        d_weights_output[d_index] = numpy.dot(
            numpy.transpose(self.joint_state),
            d_output_state[d_index + 1] * self.cell_state[s_index] * self.output_gate[s_index] * (
                        1.0 - self.output_gate[s_index]))
        d_weights_forget[d_index] = numpy.zeros((self.joint_dem, self.out_dem))
        d_weights_store[d_index] = numpy.dot(
            numpy.transpose(self.joint_state),
            d_temp * self.cell_temp[s_index] * self.store_gate[s_index] * (1.0 - self.store_gate[s_index]))
        d_weights_cell[d_index] = numpy.dot(
            numpy.transpose(self.joint_state),
            d_temp * self.store_gate[s_index] * (1.0 - (self.cell_temp[s_index] * self.cell_temp[s_index])))

        self.weights_cell += ratio * numpy.sum(d_weights_cell, 0) / learning_distance
        self.weights_forget += ratio * numpy.sum(d_weights_forget, 0) / learning_distance
        self.weights_store += ratio * numpy.sum(d_weights_store, 0) / learning_distance
        self.weights_output += ratio * numpy.sum(d_weights_output, 0) / learning_distance

        return d_input_state

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


