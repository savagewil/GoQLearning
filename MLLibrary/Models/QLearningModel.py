import numpy

from MLLibrary.Models.Model import Model
from MLLibrary.Simulations.Simulation import Simulation


class QLearningModel(Model):
    def __init__(self, data_size, controls_size, model, **kwargs):
        super().__init__(data_size, controls_size, **kwargs)
        self.model = model

        assert model.in_size == self.in_size
        assert model.out_size == self.out_size
        self.statsHandler.add_stat("Score")
        self.model.setStatsHandler(self.statsHandler)

    def predict(self, X, epsilon=0, **keyword_arguments):
        prediction = numpy.array(self.model.predict(X))
        return self.convert_prediction(prediction, epsilon=epsilon)

    @staticmethod
    def convert_prediction(prediction, epsilon=0):
        if epsilon > numpy.random.random():
            index = numpy.floor(numpy.random.random() * (len(prediction)))
            controls = numpy.zeros(numpy.shape(prediction))
            controls[index] = 1.0
            return controls
        else:
            max_value = numpy.max(prediction)
            return prediction == max_value

    def fit(self, data, batch=1,
            max_iterations=0, target_accuracy=None,
            batches_in_accuracy=1,
            err_der=(lambda Y, P: (Y - P) / 2.0),
            err=(lambda Y, P: (Y - P) ** 2.0),
            ratio=0.1, epsilon=0, gamma=1.0):
        simulation: Simulation = data
        accuracy = numpy.zeros((batches_in_accuracy))
        iteration = 0
        while (iteration < max_iterations or max_iterations <= 0) and (
                (target_accuracy is None) or any(accuracy < target_accuracy)):
            accuracy[1:] = accuracy[:-1]
            accuracy[0] = 0
            X = []
            Y = []
            for batch_index in range(batch):
                self.clear()
                X_temp, Y_temp, rewards, score = simulation.run(self)
                accuracy[0] += score
                self.statsHandler.add_to_trial("Score", score)

                q_value = numpy.zeros(numpy.shape(rewards))
                q_value[-1] = rewards[-1]

                X.append(X_temp)

                for q_index in range(numpy.shape(q_value)[0]-2, -1):
                    q_value[q_index] = rewards[q_index] + gamma * q_value[q_index + 1]

                Y.append([[(q_value[i] if v != 0.0 else None) for v in Y_temp[i, :]] for i in range(len(q_value))])

                self.clear()

            self.model.fit((X, Y), batch=batch, max_iterations=1)
            self.statsHandler.add_trial("Score")
            accuracy[0, :] = (accuracy[0, :] / batch)
            print("Score: %s" % str(accuracy))
            iteration += 1
        return accuracy

    def clear(self):
        self.model.clear()
