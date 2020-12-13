import sys

from MLLibrary.LSTMNet import LSTMNet
from random import choice
import numpy
import matplotlib.pyplot as plt

I = 1
O = 1
D = 2
LSTM = LSTMNet(I, O, D)
MAX_ITER = 1000
BATCH = 100
LEARNING_RATIO = 10.0

def get_X():
    while True:
        for b1 in [0.0, 1.0]:
            yield [[b1]]



X = get_X()
Y = get_X()

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