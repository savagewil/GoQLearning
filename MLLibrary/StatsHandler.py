from matplotlib import pyplot as plt
import numpy as np


class StatsHandler:
    def __init__(self):
        self.stats = {}

    def add_stat(self, name):
        self.stats[name] = [[]]

    def add_trial(self, name, index=-1):
        if index < 0 or index > len(self.actual):
            self.stats[name].append([])
        else:
            self.stats[name].insert(index, [])

    def add_to_trial(self, name, data, trial=-1, index=-1):
        if index < 0 or index > len(self.actual):
            self.stats[name][trial].append(data)
        else:
            self.stats[name][trial].insert(index, data)

    def add_with_trial(self, name, data, trial=-1):
        if trial < 0 or trial > len(self.actual):
            self.stats[name].append([data])
        else:
            self.stats[name].insert(trial, [data])

    def get_average_trial(self, name, trial=-1):
        stat = np.array(self.stats[name][trial])
        return np.average(stat)

    def get_median_trial(self, name, trial=-1):
        stat = np.array(self.stats[name][trial])
        return np.median(stat)

    def get_average_trials(self, name):
        stat = np.array(self.stats[name])
        return np.average(stat, 1)

    def get_median_trials(self, name):
        stat = np.array(self.stats[name])
        return np.median(stat, 1)

    def get_average(self, name):
        stat = np.array(self.stats[name]).flatten()
        return np.average(stat)

    def get_median(self, name):
        stat = np.array(self.stats[name]).flatten()
        return np.median(stat)

    def plot_average_vs_trial(self, name, scatter=False, **kwargs):
        f, ax = plt.subplots(**kwargs)
        averages = self.get_average_trials(name)
        if scatter:
            ax.scatter(np.arange(1, len(averages)+1), averages)
        else:
            ax.plot(np.arange(1, len(averages)+1), averages)
        ax.set_title('%s Averages vs Trials' % name)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Average')
        return ax

    def plot_median_vs_trial(self, name, scatter=False, **kwargs):
        f, ax = plt.subplots(**kwargs)
        medians = self.get_median_trials(name)
        if scatter:
            ax.scatter(np.arange(1, len(medians)+1), medians)
        else:
            ax.plot(np.arange(1, len(medians)+1), medians)
        ax.set_title('%s Medians vs Trials' % name)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Median')
        return ax

    def plot_stat(self, name, scatter=False, **kwargs):
        f, ax = plt.subplots(**kwargs)
        stat = np.array([s for trial in self.stats[name] for s in trial]).flatten()
        if scatter:
            ax.scatter(np.arange(1, len(stat)+1), stat)
        else:
            ax.plot(np.arange(1, len(stat)+1), stat)
        ax.set_title('%s vs Iteration' % name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('%s' % name)
        return ax
