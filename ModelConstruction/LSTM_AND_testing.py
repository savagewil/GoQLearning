from MLLibrary.LSTMNet import LSTMNet
from random import choice
import numpy
import matplotlib.pyplot as plt

I = 2
O = 1
D = 2
LSTM = LSTMNet(I, O, D)

inputs = []
error = []
accuracy = []
in_ = []
label = []
for i in range(10000):
    input = [[]]
    for j in range(I):
        input[0].append(choice([1.0])) #-1.0,
    inputs.append(input)

    if len(inputs) > 2:
        inputs = inputs[-2:]
    X = numpy.array(input)
    Y = numpy.array(1.0 if (input[0][0] == 1.0 and input[0][0] == 1.0) else 0.0) #inputs[0]
    LSTM.set_in(X)
    P = LSTM.get_out()
    d = numpy.zeros((1, 3))
    d = (Y - P)/2.0
    error.append(numpy.sum((Y - P)**2.0))
    accuracy.append(numpy.sum((Y - numpy.round(P))**2.0))
    in_.append(input[0][0])
    label.append("One" if (input[0][0]==1.0) else "Zero" )
    LSTM.learn(0.01, d)
    print("Iteration %s \t%s \t%s \t%s \t%s \t%f"%(str(i), str(input), str(Y), str(P), str(d), float(error[-1])))
    pass
iteration = numpy.arange(len(error))
in_= numpy.array(in_)
error = numpy.array(error)
window = 50
accuracy = numpy.convolve(numpy.array(accuracy), numpy.ones(window), "same") / window

ones_error = error[in_ == 1.0]
ones_accuracy = accuracy[in_ == 1.0]
ones_iteration = iteration[in_ == 1.0]

zeros_error = error[in_ != 1.0]
zeros_accuracy = accuracy[in_ != 1.0]
zeros_iteration = iteration[in_ != 1.0]

fig, ax = plt.subplots()
ax.plot(ones_iteration, ones_error, c="Blue", alpha=0.4, label="Ones")
ax.plot(zeros_iteration, zeros_error, c="Red", alpha=0.4, label="Zeros")
plt.title('LSTM Error vs Iteration for Refection')
plt.xlabel('Iteration')
plt.ylabel('Squared Error')
ax.legend()

plt.show()

fig, ax = plt.subplots()
ax.plot(ones_iteration, ones_accuracy, c="Blue", alpha=0.4, label="Ones")
ax.plot(zeros_iteration, zeros_accuracy, c="Red", alpha=0.4, label="Zeros")
plt.title('LSTM Prediction Error (rolling)  vs Iteration for Refection')
plt.xlabel('Iteration')
plt.ylabel('Rolling Error')
ax.legend()

plt.show()