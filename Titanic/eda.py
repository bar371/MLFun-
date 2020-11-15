import pandas as pd
import numpy as np
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

PATH_TRAIN = 'train.csv'
PATH_TEST = 'test.csv'

def load_data(path):
    return pd.read_csv(path)


def view_basic_data(df):
    print(ff.create_table(df.head()))
    print(df.head())


if __name__ == '__main__':
    train = load_data(PATH_TRAIN)
    view_basic_data(train)
    # test = load_data(PATH_TEST)