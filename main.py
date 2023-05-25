from sklearn.metrics import accuracy_score
from som_implementation import *

num_rows, num_cols, max_m_distance, max_learning_rate, max_steps = get_hyperparameters()
train_x, test_x, train_y, test_y = get_train_and_test_data()


def main():
    # create and train som
    _, train_x_norm = init_som()
    som = train_som()

    # collecting labels
    label_data = train_y
    _map = np.empty(shape=(num_rows, num_cols), dtype=object)

    for row in range(num_rows):
        for col in range(num_cols):
            _map[row][col] = []  # empty list to store the label

    for t in range(train_x_norm.shape[0]):
        if (t + 1) % 1000 == 0:
            print("sample data: ", t + 1)
        winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
        _map[winner[0]][winner[1]].append(label_data[t])  # label of winning neuron

    # construct label map
    label_map = generate_label_map(_map, num_rows, num_cols)
    display_map(label_map, max_steps, 10)

    data = minmax_scaler(test_x)  # normalisation
    winner_labels = []
    for t in range(data.shape[0]):
        row, col = winning_neuron(data, t, som, num_rows, num_cols)
        predicted = label_map[row][col]
        winner_labels.append(predicted)

    print("Accuracy: ", accuracy_score(test_y, np.array(winner_labels)))


if __name__ == "__main__":
    main()
