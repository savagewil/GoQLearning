import sys

import numpy
import matplotlib.pyplot as plt

from MLLibrary.Models.LinearNet import LinearNet
from MLLibrary.Models.MatrixNet import MatrixNet
from MLLibrary.Models.SequenceNet import SequenceNet
from MLLibrary.StatsHandler import StatsHandler
from MLLibrary.formulas import relu, relu_derivative

I = 2
O = 1

statsHandler = StatsHandler()
NET = SequenceNet([MatrixNet(I, I,activation_function=relu,activation_derivative=relu_derivative),
                   LinearNet(I, O)
                   ], statsHandler=statsHandler)
MAX_ITER = 1000000
BATCH = 10
LEARNING_RATIO = 0.01

MIN = 0
MAX = 7
def get_X():
    index = 1
    while True:
        index += 1
        numpy.random.seed(index)
        yield [[numpy.random.randint(MIN, MAX+1), numpy.random.randint(MIN, MAX+1)]]


def get_Y():
    index = 1
    while True:
        index += 1
        numpy.random.seed(index)
        b1 = numpy.random.randint(MIN, MAX+1)
        b2 = numpy.random.randint(MIN, MAX+1)
        yield [[(b1 + b2)]]


X = get_X()
Y = get_Y()

# for i in range(10):
#     print(next(X))
#     print(next(Y))

NET.fit((X, Y), ratio=LEARNING_RATIO, batch=BATCH, max_iterations=MAX_ITER, batches_in_accuracy=5, target_accuracy=1.0)
statsHandler.plot_stat("Error", scatter=True)
plt.show()

statsHandler.plot_stat("accuracy", scatter=True)
plt.show()

statsHandler.plot_stat("p_accuracy", scatter=True)
plt.show()
sys.exit()
