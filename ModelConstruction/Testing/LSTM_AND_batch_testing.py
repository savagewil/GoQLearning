import sys

from MLLibrary.Models.LSTMNet import LSTMNet
import matplotlib.pyplot as plt

from MLLibrary.StatsHandler import StatsHandler

I = 2
O = 1
D = 2

statsHandler = StatsHandler()

LSTM = LSTMNet(I, O, D, statsHandler=statsHandler)
MAX_ITER = 1000
BATCH = 10
LEARNING_RATIO = 0.01

def get_X():
    while True:
        for b1 in [0.0, 1.0]:
            for b2 in [0.0, 1.0]:
                yield [[b1, b2]]

def get_Y():
    while True:
        for b1 in [False, True]:
            for b2 in [False, True]:
                yield 1.0 if (b1 and b2) else 0.0


X = get_X()
Y = get_Y()


LSTM.fit(X, Y, ratio=LEARNING_RATIO, batch=BATCH, max_iterations=MAX_ITER)

statsHandler.plot_stat("Error", scatter=True)
plt.show()

statsHandler.plot_stat("accuracy", scatter=True)
plt.show()

statsHandler.plot_stat("p_accuracy", scatter=True)
plt.show()

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