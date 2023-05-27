from helper_functions import *
from starting_data import *
from sklearn.metrics import accuracy_score

num_rows, num_cols, max_m_distance, max_learning_rate, max_steps = get_hyperparameters()
train_x, test_x, train_y, test_y = get_train_and_test_data()
train_x_norm = minmax_scaler(train_x)


def init_som():
    num_dims = train_x_norm.shape[1]  # number of dimensions in the input data
    np.random.seed(40)
    som = np.random.random_sample(size=(num_rows, num_cols, num_dims))  # map construction
    return som


def train_som(som):
    for step in range(max_steps):
        if (step + 1) % 1000 == 0:
            print("Iteration: ", step + 1)  # print out the current iteration for every 1k
            display_trained_map(som, init_empty_map(), step + 1, 2)
        learning_rate, neighbourhood_range = decay(step)

        t = np.random.randint(0, high=train_x_norm.shape[0])  # random index of training data
        winner = winning_neuron(train_x_norm, t, som)
        for row in range(num_rows):
            for col in range(num_cols):
                if m_distance([row, col], winner) <= neighbourhood_range:
                    som[row][col] += learning_rate * (train_x_norm[t] - som[row][col])  # update neighbour's weight

    print("SOM training completed")
    return som


def init_empty_map():
    _map = np.empty(shape=(num_rows, num_cols), dtype=object)

    for row in range(num_rows):
        for col in range(num_cols):
            _map[row][col] = []  # empty list to store the label

    return _map


def get_filled_map(_map, som):
    for t in range(train_x_norm.shape[0]):
        if (t + 1) % 1000 == 0:
            print("sample data: ", t + 1)
        winner = winning_neuron(train_x_norm, t, som)
        _map[winner[0]][winner[1]].append(train_y[t])  # label of winning neuron

    return _map


def count_accuracy(data, som, label_map):
    winner_labels = []
    for t in range(data.shape[0]):
        row, col = winning_neuron(data, t, som)
        predicted = label_map[row][col]
        winner_labels.append(predicted)

    return accuracy_score(test_y, np.array(winner_labels))


def generate_label_map(_map):
    label_map = np.zeros(shape=(num_rows, num_cols), dtype=np.int64)
    for row, items_row in enumerate(_map):
        for col, item in enumerate(items_row):
            label_map[row][col] = 2 if len(item) <= 0 else max(item, key=item.count)

    return label_map


# Best Matching Unit search
def winning_neuron(data, t, som):
    winner = [0, 0]
    shortest_distance = np.sqrt(data.shape[1])  # initialise with max distance
    for row in range(num_rows):
        for col in range(num_cols):
            _distance = e_distance(som[row][col], data[t])
            if _distance < shortest_distance:
                shortest_distance = _distance
                winner = [row, col]
    return winner


# Learning rate and neighbourhood range calculation
def decay(step):
    coefficient = 1.0 - (np.float64(step) / max_steps)
    learning_rate = coefficient * max_learning_rate
    neighbourhood_range = ceil(coefficient * max_m_distance)
    return learning_rate, neighbourhood_range


def display_trained_map(som, __map, iteration, _time):
    _map = get_filled_map(__map, som)

    # construct label map
    label_map = generate_label_map(_map)
    display_map(label_map, iteration, _time)
