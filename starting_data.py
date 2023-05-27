import numpy as np
from numpy.ma.core import ceil
from sklearn.model_selection import train_test_split


# num_neurons = 5*np.sqrt(train_x.shape[0])
# grid_size = ceil(np.sqrt(num_neurons))

# hyperparameters
def get_hyperparameters():
    num_rows = 14
    num_cols = 14
    max_m_distance = 4
    max_learning_rate = 0.5
    max_steps = int(7.5 * 10e3)
    return num_rows, num_cols, max_m_distance, max_learning_rate, max_steps


def get_data():
    data_file = "data_banknote_authentication.txt"
    data_x = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=range(0, 4), dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=(4,), dtype=np.int64)
    return data_x, data_y


# train and test split
def get_train_and_test_data():
    data_x, data_y = get_data()
    return train_test_split(data_x, data_y, test_size=0.2, random_state=42)
