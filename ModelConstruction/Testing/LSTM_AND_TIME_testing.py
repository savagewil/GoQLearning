from MLLibrary.Models.LSTMNet import LSTMNet
from random import choice
import numpy
import matplotlib.pyplot as plt
from datetime import datetime


Iterations = 10000
S = 1
learning_rate = 1.0
LSTM = LSTMNet(S, S, S + 1)
start_time = datetime.now()

inputs = []
prediction = []
expected = []
error = []
accuracy = []
in_ = []
label = []

input = [[]]
for j in range(S):
    input[0].append(choice([0.0, 1.0]))  # -1.0,
inputs.append(input)

for i in range(Iterations):
    input = [[]]
    for j in range(S):
        input[0].append(choice([0.0, 1.0])) #-1.0,
    inputs.append(input)

    if len(inputs) > 2:
        inputs = inputs[-2:]
    X = numpy.array(input)
    Y = numpy.array(1.0 if (inputs[-1][0][0] == 1.0 and inputs[-2][0][0] == 1.0) else 0.0) #inputs[0]
    LSTM.set_in(X)
    P = LSTM.get_out()
    d = numpy.zeros((1, 3))
    d = (Y - P)/2.0
    error.append(numpy.sum((Y - P)**2.0))
    accuracy.append(numpy.sum((Y - numpy.round(P))**2.0))
    prediction.append(P)
    expected.append(Y)
    in_.append(input[0][0])
    label.append("One" if (input[0][0]==1.0) else "Zero" )
    LSTM.learn(learning_rate, d)
    if (i % (Iterations//100) == 0):
        elapsed_time = datetime.now()
        diff_time = elapsed_time - start_time
        sample_error = error[-(Iterations//100):]
        sample_accuracy = accuracy[-(Iterations//100):]
        print("Iteration %s\n"
              "Elapsed Time:    %s\n"
              "Input:           %s\n"
              "Expected Output: %s\n"
              "Prediction:      %s\n"
              "Predicted Error: %f\n"
              "Actual Error:    %f\n"
              "Average Error:    %f\n"
              "Average Actual Error:    %f\n"
              "==================================" %
              (str(i),
               str(diff_time),
               str(X),
               str(Y),
               str(P),
               float(error[-1]),
               float(accuracy[-1]),
               numpy.average(sample_error),
               numpy.average(sample_accuracy)

               ))

    pass
iteration = numpy.arange(len(error))
in_= numpy.array(in_)
error = numpy.array(error)
window = 100
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

prediction = numpy.reshape(numpy.array(prediction[-window:]), window)
expected = numpy.array(expected[-window:])
accuracy = 100 * (numpy.sum(expected == numpy.round(prediction)) / window)
print(accuracy)
