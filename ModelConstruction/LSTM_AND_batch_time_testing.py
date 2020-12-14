import sys

from MLLibrary.LSTMNet import LSTMNet
from random import choice
import numpy
import matplotlib.pyplot as plt

I = 1
O = 4
D = 4
LSTM = LSTMNet(I, O, D)
MAX_ITER = 1000000
BATCH = 1000
LEARNING_RATIO = 3.0

def get_X():
    index = 1
    while True:
        index += 1
        numpy.random.seed(index)
        yield [[numpy.random.choice([0.0, 1.0])]]


def get_Y():
    index = 1
    while True:
        numpy.random.seed(index-1)
        b1 = numpy.random.choice([0.0, 1.0])
        index += 1
        numpy.random.seed(index)
        b2 = numpy.random.choice([0.0, 1.0])
        yield [[(b1 == 1.0 and b2 == 1.0), None, None, None]]


X = get_X()
Y = get_Y()

for i in range(10):
    print(next(X))
    print(next(Y))

LSTM.fit(X, Y, ratio=LEARNING_RATIO, batch=BATCH, max_iterations=MAX_ITER)

sys.exit()


# fig, ax = plt.subplots()
# ax.plot(ones_iteration, ones_error, c="Blue", alpha=0.4, label="Ones")
# ax.plot(zeros_iteration, zeros_error, c="Red", alpha=0.4, label="Zeros")
# plt.title('LSTM Error vs Iteration for Refection')
# plt.xlabel('Iteration')
# plt.ylabel('Squared Error')
# ax.legend()
#
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(ones_iteration, ones_accuracy, c="Blue", alpha=0.4, label="Ones")
# ax.plot(zeros_iteration, zeros_accuracy, c="Red", alpha=0.4, label="Zeros")
# plt.title('LSTM Prediction Error (rolling)  vs Iteration for Refection')
# plt.xlabel('Iteration')
# plt.ylabel('Rolling Error')
# ax.legend()

# plt.show()