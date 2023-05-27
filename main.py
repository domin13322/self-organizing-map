from som_implementation import *


def main():
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # create and train som
    som = train_som(init_som())

    # collecting labels
    _map = get_filled_map(init_empty_map(), som)

    # construct label map
    label_map = generate_label_map(_map)
    display_map(label_map, max_steps, 10)

    data = minmax_scaler(test_x)  # normalisation
    accuracy = count_accuracy(data, som, label_map)
    print(f"accuracy: {accuracy}")


if __name__ == "__main__":
    main()
