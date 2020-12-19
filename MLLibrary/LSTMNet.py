import types
from abc import abstractmethod
from typing import Tuple, List, Callable, Type

import numpy, random, math, pygame

from MLLibrary.Layer import Layer
from MLLibrary.StatsHandler import StatsHandler
from MLLibrary.formulas import distance_formula, sigmoid, tanh, tanh_derivative, sigmoid_der, color_formula


class LSTMNet(Layer):

    def __init__(self, in_size: int, out_size: int, learning_distance: int = 0,
                 table_block: int = 20, **kwargs):
        # super().__init__(in_size, out_size, tanh, tanh_derivative, color_formula_param)
        super().__init__(in_size=in_size, out_size=out_size, **kwargs)
        self.in_size: int = in_size
        self.out_size: int = out_size
        self.joint_size: int = self.in_size + self.out_size + 1

        self.weights_forget = numpy.random.random((self.joint_size, self.out_size)) * 2.0 - 1.0
        self.weights_store = numpy.random.random((self.joint_size, self.out_size)) * 2.0 - 1.0
        self.weights_cell = numpy.random.random((self.joint_size, self.out_size)) * 2.0 - 1.0
        self.weights_output = numpy.random.random((self.joint_size, self.out_size)) * 2.0 - 1.0

        self.learning_distance = learning_distance
        self.table_block = table_block
        self.index = 0

        self.statsHandler.add_stat("Error")
        self.statsHandler.add_stat("p_accuracy")
        self.statsHandler.add_stat("accuracy")

        self.gradient_weights_forget = numpy.zeros((self.joint_size, self.out_size))
        self.gradient_weights_store = numpy.zeros((self.joint_size, self.out_size))
        self.gradient_weights_cell = numpy.zeros((self.joint_size, self.out_size))
        self.gradient_weights_output = numpy.zeros((self.joint_size, self.out_size))
        self.gradient_count = 0

        if self.learning_distance > 0:
            self.mem_size = self.learning_distance + 1
            self.forget_gate = numpy.zeros((self.mem_size, self.out_size))
            self.store_gate = numpy.zeros((self.mem_size, self.out_size))
            self.output_gate = numpy.zeros((self.mem_size, self.out_size))

            self.input_state = numpy.zeros((self.mem_size, self.in_size))
            self.joint_state = numpy.ones((self.mem_size, self.joint_size))

            self.cell_state = numpy.zeros((self.mem_size, self.out_size))
            self.input_tanh = numpy.zeros((self.mem_size, self.out_size))
            self.output_tanh = numpy.zeros((self.mem_size, self.out_size))
            self.output_state = numpy.zeros((self.mem_size, self.out_size))
        else:
            self.forget_gate = numpy.zeros((self.table_block, self.out_size))
            self.store_gate = numpy.zeros((self.table_block, self.out_size))
            self.output_gate = numpy.zeros((self.table_block, self.out_size))

            self.input_state = numpy.zeros((self.table_block, self.in_size))
            self.joint_state = numpy.ones((self.table_block, self.joint_size))

            self.cell_state = numpy.zeros((self.table_block, self.out_size))
            self.input_tanh = numpy.zeros((self.table_block, self.out_size))
            self.output_tanh = numpy.zeros((self.table_block, self.out_size))
            self.output_state = numpy.zeros((self.table_block, self.out_size))
            self.mem_size = self.table_block

    def clear(self):
        # self.gradient_weights_forget = numpy.zeros((self.joint_size, self.out_size))
        # self.gradient_weights_store = numpy.zeros((self.joint_size, self.out_size))
        # self.gradient_weights_cell = numpy.zeros((self.joint_size, self.out_size))
        # self.gradient_weights_output = numpy.zeros((self.joint_size, self.out_size))
        # self.gradient_count = 0

        self.index = 0

        if self.learning_distance > 0:
            self.mem_size = self.learning_distance + 1
            self.forget_gate = numpy.zeros((self.mem_size, self.out_size))
            self.store_gate = numpy.zeros((self.mem_size, self.out_size))
            self.output_gate = numpy.zeros((self.mem_size, self.out_size))

            self.input_state = numpy.zeros((self.mem_size, self.in_size))
            self.joint_state = numpy.ones((self.mem_size, self.joint_size))

            self.cell_state = numpy.zeros((self.mem_size, self.out_size))
            self.input_tanh = numpy.zeros((self.mem_size, self.out_size))
            self.output_tanh = numpy.zeros((self.mem_size, self.out_size))
            self.output_state = numpy.zeros((self.mem_size, self.out_size))
        else:
            self.forget_gate = numpy.zeros((self.table_block, self.out_size))
            self.store_gate = numpy.zeros((self.table_block, self.out_size))
            self.output_gate = numpy.zeros((self.table_block, self.out_size))

            self.input_state = numpy.zeros((self.table_block, self.in_size))
            self.joint_state = numpy.ones((self.table_block, self.joint_size))

            self.cell_state = numpy.zeros((self.table_block, self.out_size))
            self.input_tanh = numpy.zeros((self.table_block, self.out_size))
            self.output_tanh = numpy.zeros((self.table_block, self.out_size))
            self.output_state = numpy.zeros((self.table_block, self.out_size))
            self.mem_size = self.table_block

    def propagate_forward(self, X):
        self.set_in(X)
        return self.get_out()

    # def predict(self, X, **keyword_arguments):
    #     return self.propagate_forward(X)

    def propagate_backward(self, gradient):

        (g_input_state,
         g_weights_cell,
         g_weights_forget,
         g_weights_store,
         g_weights_output
         ) = self.get_gradients(gradient)

        self.gradient_weights_cell += g_weights_cell
        self.gradient_weights_forget += g_weights_forget
        self.gradient_weights_store += g_weights_store
        self.gradient_weights_output += g_weights_output
        self.gradient_count += 1

        return g_input_state

    def update_weights(self, learning_rate):
        self.weights_cell += self.gradient_weights_cell     / self.gradient_count
        self.weights_forget += self.gradient_weights_forget / self.gradient_count
        self.weights_store += self.gradient_weights_store   / self.gradient_count
        self.weights_output += self.gradient_weights_output / self.gradient_count


        self.gradient_count = 0
        self.gradient_weights_forget[:, :] = 0
        self.gradient_weights_store[:, :] = 0
        self.gradient_weights_cell[:, :] = 0
        self.gradient_weights_output[:, :] = 0

    def set_in(self, array: List[float]):
        if (1, self.in_size) == numpy.shape(array):
            self.add_new_memory()
            index_mod = [self.index % self.mem_size]
            index_before = [(self.index - 1) % self.mem_size]
            self.input_state[index_mod, :] = array
            self.joint_state[index_mod, :self.in_size] = self.input_state[index_mod, :]
            self.joint_state[index_mod, self.in_size:self.in_size + self.out_size] = self.output_state[index_before, :]

            # print(self_input_state)
        else:
            raise ValueError("Array must be (1,%d) it is %s" % (self.in_size, str(numpy.shape(array))))

    def add_new_memory(self):
        if self.learning_distance > 0:
            self.index += 1
            new_mem = (self.index + 1) % self.mem_size
            self.forget_gate[new_mem, :] = 0
            self.store_gate[new_mem, :] = 0
            self.output_gate[new_mem, :] = 0

            self.input_state[new_mem, :] = 0
            self.joint_state[new_mem, :] = 1

            self.cell_state[new_mem, :] = 0
            self.input_tanh[new_mem, :] = 0
            self.output_tanh[new_mem, :] = 0
            self.output_state[new_mem, :] = 0
        else:
            self.index += 1
            if self.index == (self.mem_size - 1):
                forget_gate_temp = self.forget_gate
                store_gate_temp = self.store_gate
                output_gate_temp = self.output_gate

                input_state_temp = self.input_state
                joint_state_temp = self.joint_state

                cell_state_temp = self.cell_state
                cell_tanh_temp = self.input_tanh
                output_tanh_temp = self.output_tanh
                output_state_temp = self.output_state

                old_mem_size = self.mem_size
                self.mem_size = self.mem_size + self.table_block

                self.forget_gate = numpy.zeros((self.mem_size, self.out_size))
                self.store_gate = numpy.zeros((self.mem_size, self.out_size))
                self.output_gate = numpy.zeros((self.mem_size, self.out_size))

                self.input_state = numpy.zeros((self.mem_size, self.in_size))
                self.joint_state = numpy.ones((self.mem_size, self.joint_size))

                self.cell_state = numpy.zeros((self.mem_size, self.out_size))
                self.input_tanh = numpy.zeros((self.mem_size, self.out_size))
                self.output_tanh = numpy.zeros((self.mem_size, self.out_size))
                self.output_state = numpy.zeros((self.mem_size, self.out_size))

                # Move old data into memory

                self.forget_gate[:old_mem_size, :] = forget_gate_temp
                self.store_gate[:old_mem_size, :] = store_gate_temp
                self.output_gate[:old_mem_size, :] = output_gate_temp

                self.input_state[:old_mem_size, :] = input_state_temp
                self.joint_state[:old_mem_size, :] = joint_state_temp

                self.cell_state[:old_mem_size, :] = cell_state_temp
                self.input_tanh[:old_mem_size, :] = cell_tanh_temp
                self.output_tanh[:old_mem_size, :] = output_tanh_temp
                self.output_state[:old_mem_size, :] = output_state_temp

    def get_out(self):
        # print("Out ran")
        index_mod = [self.index % self.mem_size]
        index_before = [(self.index - 1) % self.mem_size]

        self.forget_gate[index_mod, :] = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                           self.weights_forget))
        self.store_gate[index_mod, :] = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                          self.weights_store))
        self.output_gate[index_mod, :] = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                           self.weights_output))
        self.input_tanh[index_mod, :] = tanh(numpy.dot(self.joint_state[index_mod, :],
                                                       self.weights_cell))
        self.cell_state[index_mod, :] = numpy.add(numpy.multiply(self.cell_state[index_before, :],
                                                                 self.forget_gate[index_mod, :]),
                                                  numpy.multiply(self.input_tanh[index_mod, :],
                                                                 self.store_gate[index_mod, :]))

        self.output_tanh[index_mod, :] = tanh(self.cell_state[index_mod, :])

        self.output_state[index_mod, :] = numpy.multiply(self.output_tanh[index_mod, :],
                                                         self.output_gate[index_mod, :])

        return self.output_state[index_mod, :]

    def get_gradients(self, derivative: List[float]):
        learning_distance = self.learning_distance
        if learning_distance <= 0:
            learning_distance = self.index
        else:
            learning_distance = min(self.learning_distance, self.index)

        mem_index_mod = [self.index % self.mem_size]
        mem_index_before = [(self.index - 1) % self.mem_size]

        g_output_state = numpy.zeros((learning_distance + 1, self.out_size))
        g_input_state = numpy.zeros((learning_distance + 1, self.in_size))
        g_joint_state = numpy.zeros((learning_distance + 1, self.joint_size))
        g_cell_state = numpy.zeros((learning_distance + 1, self.out_size))

        g_weights_forget = numpy.zeros((learning_distance + 1, self.joint_size, self.out_size))
        g_weights_output = numpy.zeros((learning_distance + 1, self.joint_size, self.out_size))
        g_weights_store = numpy.zeros((learning_distance + 1, self.joint_size, self.out_size))
        g_weights_cell = numpy.zeros((learning_distance + 1, self.joint_size, self.out_size))

        g_output_state[mem_index_mod] = derivative

        # self_joint_state = self.joint_state
        # # print(self_joint_state)
        # self_input_state = self.input_state
        # # print(self_input_state)
        for index in range(0, learning_distance):
            mem_index_mod = [(self.index - index) % self.mem_size]
            mem_index_before = [(self.index - 1 - index) % self.mem_size]

            g_temp = (g_output_state[mem_index_mod, :] * self.output_gate[mem_index_mod, :] * tanh_derivative(
                self.output_tanh[mem_index_mod, :])
                      + g_cell_state[mem_index_mod, :])

            g_forget = (g_temp * self.cell_state[mem_index_before, :] * sigmoid_der(self.forget_gate[mem_index_mod, :]))
            g_store = (g_temp * self.input_tanh[mem_index_before, :] * sigmoid_der(self.store_gate[mem_index_mod, :]))
            g_cell = (g_temp * self.store_gate[mem_index_mod, :] * tanh_derivative(
                self.input_tanh[mem_index_before, :]))
            g_output = (g_output_state[mem_index_mod, :] * self.input_tanh[mem_index_mod, :] * sigmoid_der(
                self.output_gate[mem_index_mod, :]))

            g_weights_forget[mem_index_mod, :] = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_forget
            g_weights_output[mem_index_mod, :] = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_output
            g_weights_store[mem_index_mod, :] = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_store
            g_weights_cell[mem_index_mod, :] = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_cell

            g_cell_state[mem_index_before, :] = (g_temp * self.forget_gate[mem_index_mod, :])
            g_joint_state[mem_index_mod, :] = (g_forget @ numpy.transpose(self.weights_forget) +
                                               g_store @ numpy.transpose(self.weights_store) +
                                               g_cell @ numpy.transpose(self.weights_cell) +
                                               g_output @ numpy.transpose(self.weights_output))

            g_input_state[mem_index_before, :] = g_joint_state[mem_index_mod, :self.in_size]
            g_output_state[mem_index_mod, :] = g_joint_state[mem_index_mod, self.in_size:self.in_size + self.out_size]

        learning_distance = 1.0

        return (numpy.sum(g_input_state, 0) / learning_distance,
                numpy.sum(g_weights_cell, 0) / learning_distance,
                numpy.sum(g_weights_forget, 0) / learning_distance,
                numpy.sum(g_weights_store, 0) / learning_distance,
                numpy.sum(g_weights_output, 0) / learning_distance)

    def learn(self, ratio: float, gradient: List[float]):
        g_input_state = self.propagate_backward(gradient)
        self.update_weights(ratio)

        return g_input_state

