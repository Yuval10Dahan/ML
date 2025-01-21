import numpy as np
import time


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
                stored_labels.append(1)  # Assign 1 for type1
            elif label == type2:
                stored_data.append([second_feature, third_feature])
                stored_labels.append(-1)  # Assign -1 for type2

    return np.array(stored_data), np.array(stored_labels)


# Entropy calculation
def calculate_entropy(labels):

    # find the unique elements in labels and their corresponding counts
    _, counts = np.unique(labels, return_counts=True)

    # computes the probability of each label
    probabilities = counts / len(labels)

    # entropy formula
    entropy = np.sum(probabilities * np.log2(1 / (probabilities + 1e-9)))

    return entropy


# Split data based on a threshold
def split_data(data, labels, feature_index, threshold):

    # identify indexes where the feature value is less than or equal to the threshold
    left_indexes = data[:, feature_index] <= threshold

    # identify indexes where the feature value is greater than the threshold
    right_indexes = data[:, feature_index] > threshold

    return data[left_indexes], labels[left_indexes], data[right_indexes], labels[right_indexes]


# ID3 Algorithm
def id3(data, labels, depth, current_depth=0):

    # the tree has reached the maximum allowed depth, or
    # all data points in the current node have the same
    # label, so no further splitting is necessary.
    if current_depth == depth or len(np.unique(labels)) == 1:

        # assigns the class label based on the majority class.
        # If the sum of labels is positive, it assigns 1; otherwise, -1.
        leaf_label = 1 if np.sum(labels) >= 0 else -1

        # return leaf node
        return {'leaf': True, 'label': leaf_label}

    number_of_features = data.shape[1]
    best_feature, best_threshold = None, None

    # tracks the highest information gain found during the current split search
    best_info_gain = -np.inf
    best_split = None

    for feature_index in range(number_of_features):

        # extract unique values in the current feature column
        thresholds = np.unique(data[:, feature_index])
        for threshold in thresholds:

            # split the dataset based on the current feature_index and threshold
            left_data, left_labels, right_data, right_labels = split_data(data, labels, feature_index, threshold)

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            # compute the entropy of the left and right label groups
            left_entropy = calculate_entropy(left_labels)
            right_entropy = calculate_entropy(right_labels)

            # calculate information gain by known formula.
            # information gain is the reduction in entropy after splitting the data.
            info_gain = calculate_entropy(labels) - (
                (len(left_labels) / len(labels)) * left_entropy + (len(right_labels) / len(labels)) * right_entropy
            )

            # update best_split if info_gain is higher
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold
                best_split = (left_data, left_labels, right_data, right_labels)

    # case where no split is found
    if best_split is None:
        leaf_label = 1 if np.sum(labels) >= 0 else -1
        return {'leaf': True, 'label': leaf_label}

    # recursively build left and right subtrees
    left_tree = id3(best_split[0], best_split[1], depth, current_depth + 1)
    right_tree = id3(best_split[2], best_split[3], depth, current_depth + 1)

    return {
        'leaf': False,
        'feature': best_feature,
        'threshold': best_threshold,
        'left': left_tree,
        'right': right_tree
    }


def evaluate_tree(tree, data, labels):

    # generate predictions for all data points by traversing the tree
    predictions = np.array([predict(tree, x) for x in data])

    # compute the classification error
    error = np.mean(labels != predictions)

    return error


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


# Main execution
if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "iris.txt"
    label1 = "Iris-versicolor"
    label2 = "Iris-virginica"

    # load the data
    data, labels = load_data(file_path, label1, label2)

    # set maximum depth
    k = 3

    # build the tree using ID3
    id3_tree = id3(data, labels, k)

    # evaluate the tree
    error = evaluate_tree(id3_tree, data, labels)

    # print the tree
    print("Decision Tree Structure (ID3):")
    print_tree(id3_tree)

    # print the error
    print("Error (ID3):", error)

    end_time = time.perf_counter()  # End timing
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
