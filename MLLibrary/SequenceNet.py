import types
from abc import ABC
from typing import List

import numpy

from MLLibrary.Layer import Layer
from MLLibrary.formulas import distance_formula, sigmoid, tanh, tanh_derivative, sigmoid_der, color_formula


class SequenceNet(Layer):

    def __init__(self, sequence: List[Layer], **kwargs):
        super().__init__(**kwargs)
        self.in_dem = sequence[0].in_dem
        self.out_dem = sequence[-1].out_dem
        self.sequence = sequence


        self.inputs = numpy.zeros((1, self.in_dem))
        self.outputs = numpy.zeros((1, self.out_dem))

        self.statsHandler.add_stat("Error")
        self.statsHandler.add_stat("p_accuracy")
        self.statsHandler.add_stat("accuracy")

    def set_in(self, X):
        if (1, self.in_dem) == numpy.shape(X):
            self.inputs = X
        else:
            raise ValueError("Array must be (1,%d) it is %s" % (self.in_dem, str(numpy.shape(X))))

    def clear(self):
        for layer in self.sequence:
            layer.clear()

    def predict(self, X, **keyword_arguments):
        return self.propagate_forward(X)

    def propagate_forward(self, X):
        self.set_in(X)
        input_arr = self.inputs
        output_arr = []
        for layer in self.sequence:
            output_arr = layer.propagate_forward(input_arr)
            input_arr = output_arr
        self.outputs = output_arr
        return self.outputs

    def propagate_backward(self, gradient_Y):
        gradient = gradient_Y
        for layer_index in range(len(self.sequence)-1, -1, -1):
            layer = self.sequence[layer_index]
            gradient = layer.propagate_backward(gradient)
        return gradient

    def update_weights(self, learning_rate):
        for layer in self.sequence:
            layer.update_weights(learning_rate)

    # def fit(self, X, Y, ratio=0.1, batch=1,
    #         max_iterations=0, target_accuracy=1.0,
    #         err_der=(lambda Y, P: (Y - P) / 2.0),
    #         err=(lambda Y, P: (Y - P) ** 2.0)):
    #     accuracy = numpy.zeros(self.out_dem)
    #     iteration = 0
    #     while (iteration < max_iterations or max_iterations <= 0) and any(accuracy < target_accuracy):
    #         if isinstance(X, types.GeneratorType):
    #             x_data = [next(X) for _ in range(batch)]
    #         else:
    #             x_data = X[iteration * batch:(iteration + 1) * batch]
    #
    #         if isinstance(Y, types.GeneratorType):
    #             y_data = [next(Y) for _ in range(batch)]
    #         else:
    #             y_data = Y[iteration * batch:(iteration + 1) * batch]
    #
    #         accuracy = 0
    #         p_accuracy = 0
    #         for index in range(batch):
    #             # self.set_in(x_data[index])
    #             # prediction = self.get_out()
    #             prediction = self.propagate_forward(x_data[index])
    #             y_data_temp = numpy.reshape(y_data[index], self.out_dem)
    #             keep = y_data_temp != None
    #             y_data_temp[y_data_temp == None] = 0
    #
    #             t_error = err(y_data_temp, prediction)
    #             t_p_accuracy = keep * numpy.abs(y_data_temp - numpy.reshape(prediction, self.out_dem))
    #             t_accuracy = keep * numpy.abs(y_data_temp - numpy.round(numpy.reshape(prediction, self.out_dem)))
    #             p_accuracy += t_p_accuracy
    #             accuracy += t_accuracy
    #
    #             self.statsHandler.add_to_trial("Error", t_error[0][0])
    #             self.statsHandler.add_to_trial("p_accuracy", 1.0 - t_p_accuracy[0])
    #             self.statsHandler.add_to_trial("accuracy", 1.0 - t_accuracy[0])
    #
    #             gradient = keep * err_der(y_data_temp, prediction)
    #
    #             self.propagate_backward(gradient)
    #
    #         self.update_weights(ratio)
    #         self.statsHandler.add_trial("Error")
    #         self.statsHandler.add_trial("p_accuracy")
    #         self.statsHandler.add_trial("accuracy")
    #
    #         accuracy = (1.0 - accuracy / (batch * self.out_dem))
    #         p_accuracy = (1.0 - p_accuracy / (batch * self.out_dem))
    #         print("Accuracy: %s\tPredicted Accuracy: %s" % (str(accuracy), str(p_accuracy)))
    #
    #         iteration += 1
