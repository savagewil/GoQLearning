import types
from abc import ABC

import numpy

from MLLibrary.Layer import Layer
from MLLibrary.formulas import distance_formula, sigmoid, tanh, tanh_derivative, sigmoid_der, color_formula


class LinearNet(Layer):

    def __init__(self, in_size: int, out_size: int,
                 **kwargs):
        super().__init__(in_size=in_size, out_size=out_size, **kwargs)
        self.in_size = in_size
        self.out_size = out_size

        self.weights = numpy.random.random((self.in_size+1, self.out_size)) * 2.0 - 1.0
        self.gradients_weights = numpy.ones(numpy.shape(self.weights))
        self.gradient_count = 0

        self.inputs = numpy.zeros((1, self.in_size+1))
        self.outputs = numpy.zeros((1, self.out_size))

        self.statsHandler.add_stat("Error")
        self.statsHandler.add_stat("p_accuracy")
        self.statsHandler.add_stat("accuracy")

    def clear(self):
        self.inputs = numpy.zeros((1, self.in_size))
        self.outputs = numpy.zeros((1, self.out_size))

    def set_in(self, X):
        if (1, self.in_size) == numpy.shape(X):
            self.inputs[0:1, :self.in_size] = X
        else:
            raise ValueError("Array must be (1,%d) it is %s" % (self.in_size, str(numpy.shape(X))))

    def propagate_forward(self, X):
        self.set_in(X)
        self.outputs = self.inputs @ self.weights
        return self.outputs

    def propagate_backward(self, gradient):
        self.gradients_weights += numpy.transpose(self.inputs) @ gradient
        self.gradient_count += 1
        in_gradient = gradient @ numpy.transpose(self.weights)
        return in_gradient[:,:self.in_size]

    def update_weights(self, learning_rate):

        self.weights += learning_rate * self.gradients_weights / self.gradient_count

        self.gradient_count = 0
        self.gradients_weights = numpy.ones(numpy.shape(self.weights))
