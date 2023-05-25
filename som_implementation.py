from helper_functions import *
from starting_data import *

num_rows, num_cols, max_m_distance, max_learning_rate, max_steps = get_hyperparameters()
train_x, test_x, train_y, test_y = get_train_and_test_data()


def init_som():
    train_x_norm = minmax_scaler(train_x)  # normalisation
    num_dims = train_x_norm.shape[1]  # number of dimensions in the input data
    np.random.seed(40)
    som = np.random.random_sample(size=(num_rows, num_cols, num_dims))  # map construction

    return som, train_x_norm


def train_som():
    som, train_x_norm = init_som()
    for step in range(max_steps):
        if (step + 1) % 1000 == 0:
            print("Iteration: ", step + 1)  # print out the current iteration for every 1k
        learning_rate, neighbourhood_range = decay(step, max_steps, max_learning_rate, max_m_distance)

        t = np.random.randint(0, high=train_x_norm.shape[0])  # random index of training data
        winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
        for row in range(num_rows):
            for col in range(num_cols):
                if m_distance([row, col], winner) <= neighbourhood_range:
                    som[row][col] += learning_rate * (train_x_norm[t] - som[row][col])  # update neighbour's weight

    print("SOM training completed")

    return som
