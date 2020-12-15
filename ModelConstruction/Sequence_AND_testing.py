import sys

from MLLibrary.LSTMNet import LSTMNet
from random import choice
import numpy
import matplotlib.pyplot as plt

from MLLibrary.MatrixNet import MatrixNet
from MLLibrary.SequenceNet import SequenceNet

I = 2
O = 1
NET = SequenceNet([MatrixNet(I,
                             I), MatrixNet(I, O)])
MAX_ITER = 1000000
BATCH = 100
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

for i in range(10):
    print(next(X))
    print(next(Y))

NET.fit(X, Y, ratio=LEARNING_RATIO, batch=BATCH, max_iterations=MAX_ITER)

sys.exit()
