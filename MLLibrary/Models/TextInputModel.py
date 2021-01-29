from abc import ABC, abstractmethod
from typing import Dict

import numpy
from MLLibrary.Models.Model import Model
from MLLibrary.StatsHandler import StatsHandler


class TextInputModel(Model):
    def __init__(self, out_size, in_out_table: Dict, null_output=None, **kwargs):
        super().__init__(0, out_size, **kwargs)
        self.in_out_table = in_out_table
        self.null_output = null_output if null_output is not None else numpy.zeros(out_size)
        if "statsHandler" in kwargs:
            self.statsHandler = kwargs["statsHandler"]
        else:
            self.statsHandler = StatsHandler()

    def fit(self, data, batch=1,
            max_iterations=0, target_accuracy=None,
            batches_in_accuracy=1,
            err_der=(lambda Y, P: (Y - P) / 2.0),
            err=(lambda Y, P: (Y - P) ** 2.0), **keyword_arguments):
        pass

    def clear(self):
        pass

    def predict(self, X, **keyword_arguments):
        print("Data\n" + str(X) + "\nEnter input:")
        text_in = input()
        if text_in in self.in_out_table:
            return self.in_out_table[text_in]
        else:
            return self.null_output
