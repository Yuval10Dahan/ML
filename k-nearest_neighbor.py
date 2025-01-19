import numpy as np
from collections import Counter
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
                stored_labels.append(1)  # Assign 1 for `type1`
            elif label == type2:
                stored_data.append([second_feature, third_feature])
                stored_labels.append(-1)  # Assign -1 for `type2`

    return np.array(stored_data), np.array(stored_labels)


# Split the dataset into training and test sets
def train_test_split(data, labels, test_size=0.5):
    indexes = np.arange(len(data))
    np.random.shuffle(indexes)
    split_indexes = int(len(data) * test_size)
    train_indexes = indexes[:split_indexes]
    test_indexes = indexes[split_indexes:]

    return data[train_indexes], labels[train_indexes], data[test_indexes], labels[test_indexes]


# Implement k-NN classifier
def k_nn_classifier(train_data, train_labels, test_data, k, p):
    predictions = []

    # iterate over each test sample
    for query_point in test_data:

        # store the distance between the query point and
        # each point in the train data, along with the
        # label of the corresponding data point.
        distances = []

        for features, label in zip(train_data, train_labels):
            if np.isinf(p):
                # Chebyshev distance
                distance = np.max(np.abs(np.array(features) - np.array(query_point)))
            else:
                # Minkowski distance
                distance = np.sum(np.abs(np.array(features) - np.array(query_point)) ** p) ** (1 / p)

            distances.append((distance, label))

        # sort the distances list in ascending order based on the first element of each tuple
        distances.sort(key=lambda x: x[0])

        # select the k nearest neighbors from the sorted list
        k_neighbors = distances[:k]

        # extract the labels of the k nearest neighbors
        k_labels = [label for _, label in k_neighbors]

        # determine the most frequent label among the k nearest neighbors
        most_common = Counter(k_labels).most_common(1)

        # append the predicted label for this query point
        predictions.append(most_common[0][0])

    return np.array(predictions)


# Evaluate classifier over multiple repetitions
def evaluate_knn(file_path, type1, type2, k_list, p_list, repetitions=100):
    data, labels = load_data(file_path, type1, type2)
    results = []

    for p in p_list:
        for k in k_list:
            train_error_list = []
            test_error_list = []

            for _ in range(repetitions):
                train_data, train_labels, test_data, test_labels = train_test_split(data, labels)

                # predict labels for the training set
                train_predictions = k_nn_classifier(train_data, train_labels, train_data, k, p)

                # calculate the error on the training set
                train_error = np.mean(train_predictions != train_labels)

                # predict labels for the test set
                test_predictions = k_nn_classifier(train_data, train_labels, test_data, k, p)

                # calculate the error on the test set
                test_error = np.mean(test_predictions != test_labels)

                # store the errors
                train_error_list.append(train_error)
                test_error_list.append(test_error)

            # compute average errors and their difference
            avg_empirical_errors = np.mean(train_error_list)
            avg_true_errors = np.mean(test_error_list)
            error_difference = avg_true_errors - avg_empirical_errors

            # Append the results
            results.append({
                'p': p,
                'k': k,
                'avg_empirical_error': avg_empirical_errors,
                'avg_true_error': avg_true_errors,
                'error_difference': error_difference
            })

    return results


# Main function to execute the evaluation
if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "iris.txt"
    label1 = "Iris-versicolor"
    label2 = "Iris-virginica"
    k_list = [1, 3, 5, 7, 9]
    p_list = [1, 2, np.inf]

    results = evaluate_knn(file_path, label1, label2, k_list, p_list)

    # print results
    for result in results:
        print(f"p = {result['p']}, k = {result['k']}: "
              f"Avg empirical Error = {result['avg_empirical_error']:.4f}, "
              f"Avg true Error = {result['avg_true_error']:.4f}, "
              f"Error difference = {result['error_difference']:.4f}")

    end_time = time.perf_counter()  # End timing
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
