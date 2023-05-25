# Helper functions
import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler  # normalisation
import matplotlib.pyplot as plt
from matplotlib import colors


# Data Normalisation
def minmax_scaler(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled


# Euclidean distance
def e_distance(x, y):
    return distance.euclidean(x, y)


# Manhattan distance
def m_distance(x, y):
    return distance.cityblock(x, y)


# Best Matching Unit search
def winning_neuron(data, t, som, num_rows, num_cols):
    winner = [0, 0]
    shortest_distance = np.sqrt(data.shape[1])  # initialise with max distance
    for row in range(num_rows):
        for col in range(num_cols):
            distance = e_distance(som[row][col], data[t])
            if distance < shortest_distance:
                shortest_distance = distance
                winner = [row, col]
    return winner


# Learning rate and neighbourhood range calculation
def decay(step, max_steps, max_learning_rate, max_m_dsitance):
    coefficient = 1.0 - (np.float64(step) / max_steps)
    learning_rate = coefficient * max_learning_rate
    neighbourhood_range = ceil(coefficient * max_m_dsitance)
    return learning_rate, neighbourhood_range


def generate_label_map(_map, num_rows, num_cols):
    label_map = np.zeros(shape=(num_rows, num_cols), dtype=np.int64)
    for row, items_row in enumerate(_map):
        for col, item in enumerate(items_row):
            label_map[row][col] = 2 if len(item) <= 0 else max(item, key=item.count)

    return label_map


def display_map(label_map, iteration, _time):
    cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
    plt.imshow(label_map, cmap=cmap)
    plt.colorbar()
    plt.title(f"Iteration {str(iteration)}")
    plt.pause(_time)
    plt.clf()
