import numpy as np
import imageio
import collections
import numpy as np
import pandas as pd
from pandas import plotting

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff

# for path
import os
print(os.listdir('C:/Users/x3yusk/PycharmProjects/algorithmtryouts/MLFun/Kmeans/'))

ECULID = 1
MANHATAN = 2


class Kmeans:

    def __init__(self):
        self.init_centroies = None

    def distance_factory(self, method, sample, init_centroies):
        if method == ECULID:
            return int(np.argmin(
                [np.sqrt(np.sum([np.square(sample[i] - c[i]) for i in range(len(sample))])) for c in init_centroies]))
        elif method == MANHATAN:
            return int(np.argmin([np.sum([np.abs(sample - c)]) for c in init_centroies]))

    def train(self, k, X_train, iters=30):
        self.init_centroies = list(X_train.sample(n=k).values)
        for index in range(iters + 1):
            cen_choices = [[] for i in range(k)]
            cur_preds = []
            for sample in X_train.values:
                choice = self.distance_factory(1, sample, self.init_centroies)
                cen_choices[choice].append(sample)
                cur_preds.append(choice)
            for i, cen in enumerate(cen_choices):
                cen = np.asarray(cen)
                for j in range(len(cen[0])):
                    self.init_centroies[i][j] = np.average(cen[:, j])
            # cluster_plot(index, cur_preds, self.init_centroies)
        #
        # with imageio.get_writer("test2.gif", mode='I') as writer:
        #     for image_index in list(range(index)):
        #         writer.append_data(imageio.imread("{i}.jpg".format(i=image_index)))

    def predict(self, X_test):
        if not self.init_centroies:
            print('You need to train your model first')
        return [self.distance_factory(1, sample, self.init_centroies) for sample in X_test]

    def check_accuracy(self, preds, y_test):
        assert (len(preds) == len(y_test))
        print(np.sum([1 for p, y in zip(preds, y_test) if p == y]) / len(preds))


def cluster_plot(i,y_means, cen_centroieds):
    y_means = np.asarray(y_means)
    plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')
    plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')
    plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')
    plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
    plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')
    x1, y1 = zip(*(cen_centroieds))
    plt.scatter(x1 , y1, s = 50, c = 'blue' , label = 'centeroid')

    plt.style.use('fivethirtyeight')
    plt.title('K Means Clustering', fontsize = 20)
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.savefig("{i}.jpg".format(i=i))
    plt.show()