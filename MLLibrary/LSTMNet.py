import types
from abc import abstractmethod
from typing import Tuple, List, Callable, Type

import numpy, random, math, pygame

from MLLibrary.Layer import Layer
from MLLibrary.formulas import distance_formula, sigmoid, tanh, tanh_derivative, sigmoid_der, color_formula


class LSTMNet(Layer):
    def __init__(self, in_dem: int,
                 out_dem: int,
                 learning_distance: int = 0,
                 color_formula_param: Callable = color_formula,
                 table_block: int = 20):
        # super().__init__(in_dem, out_dem, tanh, tanh_derivative, color_formula_param)
        self.in_dem: int        = in_dem
        self.out_dem: int       = out_dem
        self.joint_dem: int     = self.in_dem + self.out_dem + 1

        self.weights_forget     = numpy.random.random((self.joint_dem, self.out_dem)) * 2.0 - 1.0
        self.weights_store      = numpy.random.random((self.joint_dem, self.out_dem)) * 2.0 - 1.0
        self.weights_cell       = numpy.random.random((self.joint_dem, self.out_dem)) * 2.0 - 1.0
        self.weights_output     = numpy.random.random((self.joint_dem, self.out_dem)) * 2.0 - 1.0

        self.gradient_weights_forget    = numpy.zeros((self.joint_dem, self.out_dem))
        self.gradient_weights_store     = numpy.zeros((self.joint_dem, self.out_dem))
        self.gradient_weights_cell      = numpy.zeros((self.joint_dem, self.out_dem))
        self.gradient_weights_output    = numpy.zeros((self.joint_dem, self.out_dem))

        self.learning_distance  = learning_distance
        self.table_block        = table_block
        self.index              = 0

        if learning_distance > 0:
            self.mem_size = learning_distance + 1
            self.forget_gate    = numpy.zeros((self.mem_size, self.out_dem))
            self.store_gate     = numpy.zeros((self.mem_size, self.out_dem))
            self.output_gate    = numpy.zeros((self.mem_size, self.out_dem))

            self.input_state    = numpy.zeros((self.mem_size, self.in_dem))
            self.joint_state    = numpy.ones((self.mem_size, self.joint_dem))


            self.cell_state     = numpy.zeros((self.mem_size, self.out_dem))
            self.input_tanh      = numpy.zeros((self.mem_size, self.out_dem))
            self.output_tanh    = numpy.zeros((self.mem_size, self.out_dem))
            self.output_state   = numpy.zeros((self.mem_size, self.out_dem))
        else:
            self.forget_gate    = numpy.zeros((table_block, self.out_dem))
            self.store_gate     = numpy.zeros((table_block, self.out_dem))
            self.output_gate    = numpy.zeros((table_block, self.out_dem))

            self.input_state    = numpy.zeros((table_block, self.in_dem))
            self.joint_state    = numpy.ones((table_block, self.joint_dem))

            self.cell_state     = numpy.zeros((table_block, self.out_dem))
            self.input_tanh      = numpy.zeros((table_block, self.out_dem))
            self.output_tanh    = numpy.zeros((table_block, self.out_dem))
            self.output_state   = numpy.zeros((table_block, self.out_dem))
            self.mem_size = table_block

        self.score: float = 0

        self.color_formula: Callable = color_formula_param


    def propagate_forward(self, X):
        self.set_in(X)
        return self.get_out()

    def predict(self, X, **keyword_arguments):
        return self.propagate_forward(X)


    def propagate_backward(self, gradient_Y):

        (g_input_state,
         g_weights_cell,
         g_weights_forget,
         g_weights_store,
         g_weights_output
         ) = self.get_gradients(gradient_Y)

        self.gradient_weights_cell   += g_weights_cell
        self.gradient_weights_forget += g_weights_forget
        self.gradient_weights_store  += g_weights_store
        self.gradient_weights_output += g_weights_output

        return g_input_state

    def update_weights(self, learning_rate):
        self.weights_cell   += self.gradient_weights_cell
        self.weights_forget += self.gradient_weights_forget
        self.weights_store  += self.gradient_weights_store
        self.weights_output += self.gradient_weights_output

        self.gradient_weights_forget[:, :]    = 0
        self.gradient_weights_store[:, :]     = 0
        self.gradient_weights_cell[:, :]      = 0
        self.gradient_weights_output[:, :]    = 0

    def set_in(self, array: List[float]):
        if (1, self.in_dem) == numpy.shape(array):
            self.add_new_memory()
            index_mod = [self.index % self.mem_size]
            index_before = [(self.index - 1) % self.mem_size]
            self.input_state[index_mod, :] = array
            self.joint_state[index_mod, :self.in_dem] = self.input_state[index_mod, :]
            self.joint_state[index_mod, self.in_dem:self.in_dem+self.out_dem] = self.output_state[index_before, :]

            # print(self_input_state)
        else:
            raise ValueError("Array must be (1,%d) it is %s" % (self.in_dem, str(numpy.shape(array))))

    def add_new_memory(self):
        if self.learning_distance > 0:
            self.index += 1
            new_mem = (self.index + 1) % self.mem_size
            self.forget_gate[new_mem, :]    = 0
            self.store_gate[new_mem, :]     = 0
            self.output_gate[new_mem, :]    = 0

            self.input_state[new_mem, :]    = 0
            self.joint_state[new_mem, :]    = 1

            self.cell_state[new_mem, :]     = 0
            self.input_tanh[new_mem, :]      = 0
            self.output_tanh[new_mem, :]    = 0
            self.output_state[new_mem, :]   = 0
        else:
            self.index += 1
            if self.index == (self.mem_size - 1):
                forget_gate_temp    = self.forget_gate
                store_gate_temp     = self.store_gate
                output_gate_temp    = self.output_gate

                input_state_temp    = self.input_state
                joint_state_temp    = self.joint_state

                cell_state_temp     = self.cell_state
                cell_tanh_temp      = self.input_tanh
                output_tanh_temp    = self.output_tanh
                output_state_temp   = self.output_state

                old_mem_size        = self.mem_size
                self.mem_size       = self.mem_size + self.table_block

                self.forget_gate    = numpy.zeros((self.mem_size, self.out_dem))
                self.store_gate     = numpy.zeros((self.mem_size, self.out_dem))
                self.output_gate    = numpy.zeros((self.mem_size, self.out_dem))

                self.input_state    = numpy.zeros((self.mem_size, self.in_dem))
                self.joint_state    = numpy.ones((self.mem_size, self.joint_dem))

                self.cell_state     = numpy.zeros((self.mem_size, self.out_dem))
                self.input_tanh      = numpy.zeros((self.mem_size, self.out_dem))
                self.output_tanh    = numpy.zeros((self.mem_size, self.out_dem))
                self.output_state   = numpy.zeros((self.mem_size, self.out_dem))

                # Move old data into memory

                self.forget_gate[:old_mem_size, :]  = forget_gate_temp
                self.store_gate[:old_mem_size, :]   = store_gate_temp
                self.output_gate[:old_mem_size, :]  = output_gate_temp

                self.input_state[:old_mem_size, :]  = input_state_temp
                self.joint_state[:old_mem_size, :]  = joint_state_temp

                self.cell_state[:old_mem_size, :]   = cell_state_temp
                self.input_tanh[:old_mem_size, :]    = cell_tanh_temp
                self.output_tanh[:old_mem_size, :]  = output_tanh_temp
                self.output_state[:old_mem_size, :] = output_state_temp

    def get_out(self):
        # print("Out ran")
        index_mod = [self.index % self.mem_size]
        index_before = [(self.index - 1) % self.mem_size]

        self.forget_gate[index_mod, :]  = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                            self.weights_forget))
        self.store_gate[index_mod, :]   = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                            self.weights_store))
        self.output_gate[index_mod, :]  = sigmoid(numpy.dot(self.joint_state[index_mod, :],
                                                            self.weights_output))
        self.input_tanh[index_mod, :]    = tanh(numpy.dot(self.joint_state[index_mod, :],
                                                          self.weights_cell))
        self.cell_state[index_mod, :]   = numpy.add(numpy.multiply(self.cell_state[index_before, :],
                                                                   self.forget_gate[index_mod, :]),
                                                    numpy.multiply(self.input_tanh[index_mod, :],
                                                                   self.store_gate[index_mod, :]))

        self.output_tanh[index_mod, :]  = tanh(self.cell_state[index_mod, :])

        self.output_state[index_mod, :] = numpy.multiply(self.output_tanh[index_mod, :],
                                                         self.output_gate[index_mod, :])

        return self.output_state[index_mod, :]

    def get_gradients(self, derivative: List[float]):
        learning_distance = self.learning_distance
        if learning_distance <= 0:
            learning_distance = self.index
        else:
            learning_distance = min(self.learning_distance, self.index)

        mem_index_mod       = [self.index % self.mem_size]
        mem_index_before    = [(self.index - 1) % self.mem_size]

        g_output_state      = numpy.zeros((learning_distance + 1, self.out_dem))
        g_input_state       = numpy.zeros((learning_distance + 1, self.in_dem))
        g_joint_state       = numpy.zeros((learning_distance + 1, self.joint_dem))
        g_cell_state        = numpy.zeros((learning_distance + 1, self.out_dem))

        g_weights_forget    = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        g_weights_output    = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        g_weights_store     = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))
        g_weights_cell      = numpy.zeros((learning_distance + 1, self.joint_dem, self.out_dem))

        g_output_state[mem_index_mod] = derivative

        # self_joint_state = self.joint_state
        # # print(self_joint_state)
        # self_input_state = self.input_state
        # # print(self_input_state)
        for index in range(0, learning_distance):
            mem_index_mod       = [(self.index - index) % self.mem_size]
            mem_index_before    = [(self.index - 1 - index) % self.mem_size]

            g_temp = (g_output_state[mem_index_mod, :] * self.output_gate[mem_index_mod, :] * tanh_derivative(self.output_tanh[mem_index_mod, :])
                      + g_cell_state[mem_index_mod, :])

            g_forget    = (g_temp * self.cell_state[mem_index_before, :] * sigmoid_der(self.forget_gate[mem_index_mod, :]))
            g_store     = (g_temp * self.input_tanh[mem_index_before, :] * sigmoid_der(self.store_gate[mem_index_mod, :]))
            g_cell      = (g_temp * self.store_gate[mem_index_mod, :] * tanh_derivative(self.input_tanh[mem_index_before, :]))
            g_output    = (g_output_state[mem_index_mod, :] * self.input_tanh[mem_index_mod, :] * sigmoid_der(self.output_gate[mem_index_mod, :]))

            g_weights_forget[mem_index_mod, :]    = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_forget
            g_weights_output[mem_index_mod, :]    = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_output
            g_weights_store[mem_index_mod, :]     = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_store
            g_weights_cell[mem_index_mod, :]      = numpy.transpose(self.joint_state[mem_index_mod, :]) @ g_cell

            g_cell_state[mem_index_before, :]   = (g_temp * self.forget_gate[mem_index_mod, :])
            g_joint_state[mem_index_mod, :]     = (g_forget @ numpy.transpose(self.weights_forget) +
                                                   g_store @ numpy.transpose(self.weights_store) +
                                                   g_cell @ numpy.transpose(self.weights_cell) +
                                                   g_output @ numpy.transpose(self.weights_output))

            g_input_state[mem_index_before, :] = g_joint_state[mem_index_mod, :self.in_dem]
            g_output_state[mem_index_mod, :]     = g_joint_state[mem_index_mod, self.in_dem:self.in_dem + self.out_dem]

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

    def fit(self, X, Y, ratio=0.1, batch=1, max_iterations=0, target_accuracy=1.0, err_der=(lambda Y, P: (Y - P)/2.0)):
        accuracy = numpy.zeros(self.out_dem)
        iteration = 0
        while (iteration < max_iterations or max_iterations <= 0) and any(accuracy < target_accuracy):
            if isinstance(X, types.GeneratorType):
                x_data = [next(X) for _ in range(batch)]
            else:
                x_data = X[iteration * batch:(iteration + 1) * batch]

            if isinstance(Y, types.GeneratorType):
                y_data = [next(Y) for _ in range(batch)]
            else:
                y_data = Y[iteration * batch:(iteration + 1) * batch]

            accuracy    = 0
            p_accuracy  = 0
            for index in range(batch):
                # self.set_in()
                # prediction = self.get_out()
                prediction = self.propagate_forward(x_data[index])
                y_data_temp = numpy.reshape(y_data[index], self.out_dem)
                keep = y_data_temp != None
                y_data_temp[y_data_temp == None] = 0
                p_accuracy += keep * numpy.abs(y_data_temp -
                                      numpy.reshape(prediction, self.out_dem))
                accuracy += keep * numpy.abs(y_data_temp -
                                      numpy.round(numpy.reshape(prediction, self.out_dem)))
                gradient = keep * err_der(y_data_temp, prediction)

                self.propagate_backward(gradient)

            self.update_weights(ratio)

            accuracy = (1.0 - accuracy / (batch * self.out_dem))
            p_accuracy = (1.0 - p_accuracy / (batch * self.out_dem))
            print("Accuracy: %s\tPredicted Accuracy: %s"%(str(accuracy),str(p_accuracy)))

            iteration += 1

