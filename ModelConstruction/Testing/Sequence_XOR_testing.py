import sys

from MLLibrary.LSTMNet import LSTMNet
from random import choice
import numpy
import matplotlib.pyplot as plt

from MLLibrary.MatrixNet import MatrixNet
from MLLibrary.SequenceNet import SequenceNet
from MLLibrary.StatsHandler import StatsHandler

I = 2
O = 1

statsHandler = StatsHandler()
NET = SequenceNet([MatrixNet(I,
                             I), MatrixNet(I, O)],statsHandler=statsHandler)
MAX_ITER = 1000000
BATCH = 50
LEARNING_RATIO = 0.1


def get_X():
    index = 1
    while True:
        index += 1
        numpy.random.seed(index)
        yield [[numpy.random.choice([0.0, 1.0]), numpy.random.choice([0.0, 1.0])]]


def get_Y():
    index = 1
    while True:
        index += 1
        numpy.random.seed(index)
        b1 = numpy.random.choice([0.0, 1.0])
        b2 = numpy.random.choice([0.0, 1.0])
        yield [[(b1 != b2)]]


X = get_X()
Y = get_Y()


NET.fit(X, Y, ratio=LEARNING_RATIO,
        batch=BATCH,
        max_iterations=MAX_ITER,
        batches_in_accuracy=5)
statsHandler.plot_stat("Error", scatter=True)
plt.show()

statsHandler.plot_average_vs_trial("accuracy", scatter=True)
plt.show()

statsHandler.plot_stat("p_accuracy", scatter=True)
plt.show()
sys.exit()
