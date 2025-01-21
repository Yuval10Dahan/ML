import numpy as np
import time
from itertools import product


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
def split_data(data, labels, feature_index, threshold):

    # identify indexes where the feature value is less than or equal to the threshold
    left_indexes = data[:, feature_index] <= threshold

    # identify indexes where the feature value is greater than the threshold
    right_indexes = data[:, feature_index] > threshold

    return data[left_indexes], labels[left_indexes], data[right_indexes], labels[right_indexes]


# Brute-force method
def generate_trees(data, labels, depth, current_depth=0):

    # the tree has reached the maximum allowed depth, or
    # all data points in the current node have the same
    # label, so no further splitting is necessary.
    if current_depth == depth or len(np.unique(labels)) == 1:

        # assigns the class label based on the majority class.
        # If the sum of labels is positive, it assigns 1; otherwise, -1.
        leaf_label = 1 if np.sum(labels) >= 0 else -1

        # return leaf node
        return [{'leaf': True, 'label': leaf_label}]

    number_of_features_in_data = data.shape[1]

    # list of unique thresholds for each feature
    thresholds_per_feature = [np.unique(data[:, f]) for f in range(number_of_features_in_data)]

    possible_trees = []

    for feature_index, thresholds in enumerate(thresholds_per_feature):
        for threshold in thresholds:

            # split the dataset based on the current feature_index and threshold
            left_data, left_labels, right_data, right_labels = split_data(data, labels, feature_index, threshold)

            # recursively generates all possible trees for the left and right subsets
            left_subtrees = generate_trees(left_data, left_labels, depth, current_depth + 1)
            right_subtrees = generate_trees(right_data, right_labels, depth, current_depth + 1)

            # iterates over all combinations(Cartesian product) of
            # left and right subtrees generated from the current split
            for left_tree, right_tree in product(left_subtrees, right_subtrees):

                # constructs a new tree and adds it to the list
                possible_trees.append({
                    'leaf': False,
                    'feature': feature_index,
                    'threshold': threshold,
                    'left': left_tree,
                    'right': right_tree
                })

    return possible_trees


def evaluate_tree(tree, data, labels):

    # generate predictions for all data points by traversing the tree
    predictions = np.array([predict(tree, x) for x in data])

    # compute the classification error
    brute_force_error = np.mean(labels != predictions)

    return brute_force_error


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
    start_time = time.perf_counter()  # Start timing

    file_path = "iris.txt"
    label1 = "Iris-versicolor"
    label2 = "Iris-virginica"

    # load the data
    data, labels = load_data(file_path, label1, label2)

    # construct all possible trees of k levels
    k = 3
    possible_trees = generate_trees(data, labels, k)

    print("size of possible trees: ", len(possible_trees))

    # evaluate all trees and find the best one
    best_tree = None
    best_error = np.inf

    for tree in possible_trees:
        error = evaluate_tree(tree, data, labels)
        if error < best_error:
            best_error = error
            best_tree = tree

    # print the tree
    print("Best Decision Tree Structure:")
    print_tree(best_tree)

    # print the error
    print("Best Error:", best_error)

    end_time = time.perf_counter()  # End timing
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
