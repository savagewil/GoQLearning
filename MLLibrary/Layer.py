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


    def predict(self, X, **keyword_arguments):
        return self.propagate_forward(X)

    def fit(self, data, ratio=0.1, batch=1,
            max_iterations=0, target_accuracy=None,
            err_der=(lambda Y, P: (Y - P) / 2.0),
            err=(lambda Y, P: (Y - P) ** 2.0),
            batches_in_accuracy=1):
        X, Y = data
        accuracy = numpy.zeros((batches_in_accuracy, self.out_size))
        p_accuracy = numpy.zeros((batches_in_accuracy, self.out_size))
        iteration = 0
        while (iteration < max_iterations or max_iterations <= 0) and ((target_accuracy is None) or any(accuracy < target_accuracy)):
            accuracy[1:, :] = accuracy[:-1, :]
            p_accuracy[1:, :] = p_accuracy[:-1, :]
            accuracy[0, :] = 0
            p_accuracy[0, :] = 0

            if isinstance(X, types.GeneratorType):
                x_data = [next(X) for _ in range(batch)]
            else:
                x_data = X[iteration * batch:(iteration + 1) * batch]

            if isinstance(Y, types.GeneratorType):
                y_data = [next(Y) for _ in range(batch)]
            else:
                y_data = Y[iteration * batch:(iteration + 1) * batch]

            for index in range(batch):
                prediction = self.propagate_forward(x_data[index])
                y_data_temp = numpy.reshape(y_data[index], self.out_size)
                keep = y_data_temp != None
                y_data_temp[y_data_temp == None] = 0
                t_error = err(y_data_temp, prediction)
                t_p_accuracy = keep * numpy.abs(y_data_temp - numpy.reshape(prediction, self.out_size))
                t_accuracy = keep * numpy.abs(y_data_temp - numpy.round(numpy.reshape(prediction, self.out_size)))
                p_accuracy[0,:] += t_p_accuracy
                accuracy[0,:] += t_accuracy
                self.statsHandler.add_to_trial("Error", t_error[0][0])
                self.statsHandler.add_to_trial("p_accuracy", 1.0 - t_p_accuracy[0]/ (batch * self.out_size))
                self.statsHandler.add_to_trial("accuracy", 1.0 - t_accuracy[0]/ (batch * self.out_size))
                gradient = keep * err_der(y_data_temp, prediction)

                self.propagate_backward(gradient)

            self.update_weights(ratio)
            self.statsHandler.add_trial("Error")
            self.statsHandler.add_trial("p_accuracy")
            self.statsHandler.add_trial("accuracy")
            accuracy[0,:] = (1.0 - accuracy[0,:] / (batch * self.out_size))
            p_accuracy[0,:] = (1.0 - p_accuracy[0,:] / (batch * self.out_size))
            print("Accuracy: %s\nPredicted Accuracy: %s" % (str(accuracy), str(p_accuracy)))
            iteration += 1

        return accuracy
