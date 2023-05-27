import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colors


# Data Normalisation
def minmax_scaler(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled


def e_distance(x, y):
    return distance.euclidean(x, y)


def m_distance(x, y):
    return distance.cityblock(x, y)


def display_map(label_map, iteration, _time):
    cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
    plt.imshow(label_map, cmap=cmap)
    plt.colorbar()
    plt.title(f"Iteration {str(iteration)}")
    plt.pause(_time)
    plt.clf()
