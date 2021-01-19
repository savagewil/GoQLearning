import sys

from MLLibrary.Models.LSTMNet import LSTMNet
import numpy

I = 1
O = 1
D = 2
LSTM = LSTMNet(I, O, D)
MAX_ITER = 1000
BATCH = 100
LEARNING_RATIO = 0.01

def get_X():
    index = 0
    while True:
        index += 1
        numpy.random.seed(index)
        yield [[numpy.random.choice([0.0, 1.0])]]


X = get_X()
Y = get_X()
next(X)

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