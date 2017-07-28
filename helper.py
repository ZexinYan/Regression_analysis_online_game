import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Helper:
    def plot_correlation_heap(self, data, target=None, title=None):
        if target:
            correlations = data.corr()[target].sort_values(ascending=False)
            plt.figure(figsize=(40, 6))
            correlations.drop(target).plot.bar()
            pd.set_option('precision', 2)
            plt.xticks(rotation=90, fontsize=7)
            plt.yticks(rotation=360)
            plt.suptitle(title)
            plt.show()
        else:
            pd.set_option('precision', 2)
            plt.figure(figsize=(20, 6))
            sns.heatmap(data.corr(), square=True)
            plt.xticks(rotation=90)
            plt.yticks(rotation=360)
            plt.suptitle("The Correlation Heatmap")
            plt.show()

    def construct_dict(self, data, col, thresholds):
        data_dummy = pd.get_dummies(data[col], '')
        data_dummy = pd.DataFrame(pd.concat([data_dummy, data[u'评论']], axis=1))
        corr = data_dummy.corr()[u'评论'].sort_values(ascending=False)
        dict = {}
        size = len(thresholds)
        for i in range(len(corr.keys())):
            value = corr.values[i]
            key = corr.keys()[i][1:]
            for i in range(size):
                if value > thresholds[i]:
                    dict[key] = size - i
                    break
        return dict
