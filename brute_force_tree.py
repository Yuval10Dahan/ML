import numpy as np


def load_data(file_path, type1, type2):
    stored_data = []  # store the data from the file
    stored_labels = []  # store the class labels

    with open(file_path, "r") as file:
        for line in file:
            # Removes leading whitespace
            line = line.strip()
            if not line:
                continue  # Skip blank lines

            # Split the line into parts
            parts = line.split()
            second_feature, third_feature = float(parts[1]), float(parts[2])
            label = parts[-1]

            if label == type1:
                stored_data.append([second_feature, third_feature])
                stored_labels.append(1)  # Assign 1 for `type1`
            elif label == type2:
                stored_data.append([second_feature, third_feature])
                stored_labels.append(-1)  # Assign -1 for `type2`

    return np.array(stored_data), np.array(stored_labels)


# Split data based on a threshold
def split_data(data, labels, feature_idx, threshold):

    # identify indexes where the feature value is less than or equal to the threshold
    left_indexes = data[:, feature_idx] <= threshold

    # identify indexes where the feature value is greater than the threshold
    right_indexes = data[:, feature_idx] > threshold

    return data[left_indexes], labels[left_indexes], data[right_indexes], labels[right_indexes]


# Brute-force method
def brute_force_tree(data, labels, k, current_level=0):

    # the tree has reached the maximum allowed depth, or
    # all data points in the current node have the same
    # label, so no further splitting is necessary.
    if current_level == k or len(np.unique(labels)) == 1:

        # assigns the class label based on the majority class.
        # If the sum of labels is positive, it assigns 1; otherwise, -1.
        leaf_label = 1 if np.sum(labels) >= 0 else -1

        # Return leaf node
        return {'leaf': True, 'label': leaf_label}

    best_split = None
    best_error = np.inf
    number_of_features_in_data = data.shape[1]

    # try all splits
    for feature_index in range(number_of_features_in_data):

        # stores each unique value in the selected feature as a potential thresholds
        thresholds = np.unique(data[:, feature_index])

        for threshold in thresholds:

            # split the dataset based on the current feature_index and threshold
            _, left_labels, _, right_labels = split_data(data, labels, feature_index, threshold)

            # misclassified points in the left and right subsets
            miss_points_left = left_labels[left_labels != 1]
            miss_points_right = right_labels[right_labels != -1]

            # total misclassification error for the current split
            error = len(miss_points_left) + len(miss_points_right)

            if error < best_error:
                best_error = error
                best_split = (feature_index, threshold)

    # no valid split is found - act as in the base case
    if best_split is None:
        leaf_label = 1 if np.sum(labels) >= 0 else -1
        return {'leaf': True, 'label': leaf_label}

    best_index, best_threshold = best_split
    left_data, left_labels, right_data, right_labels = split_data(data, labels, best_index, best_threshold)

    # constructs the current node and recursively builds the left and right subtrees
    return {
        'leaf': False,
        'feature': best_index,
        'threshold': best_threshold,
        'left': brute_force_tree(left_data, left_labels, k, current_level + 1),
        'right': brute_force_tree(right_data, right_labels, k, current_level + 1)
    }


# Prediction function
def predict(tree, x):

    # if the current node is a leaf
    if tree['leaf']:
        return tree['label']

    # compares the feature value to the threshold stored in the current node
    if x[tree['feature']] <= tree['threshold']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)


def print_tree(tree, level=0, side="Root"):
    indent = "   " * level + ("|-- " if level > 0 else "")
    if tree['leaf']:
        print(f"{indent}{side} [Leaf] Label: {tree['label']}")
    else:
        print(f"{indent}{side} [Node] Feature: {tree['feature']}, Threshold: {tree['threshold']}")
        print_tree(tree['left'], level + 1, "Left")
        print_tree(tree['right'], level + 1, "Right")


if __name__ == "__main__":

    file_path = "iris.txt"
    label1 = "Iris-versicolor"
    label2 = "Iris-virginica"

    # load the data
    data, labels = load_data(file_path, label1, label2)

    # build brute-force tree
    k = 3
    brute_force = brute_force_tree(data, labels, k)

    # print the tree
    print("Decision Tree Structure:")
    print_tree(brute_force)

    # predictions
    brute_force_predictions = np.array([predict(brute_force, x) for x in data])

    # compute error
    brute_force_error = np.mean(labels != brute_force_predictions)
    print("Brute Force Error:", brute_force_error)
