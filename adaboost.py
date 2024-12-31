import numpy as np
import random

# User input for labels
label1 = "Iris-setosa"
label2 = "Iris-versicolor"
label3 = "Iris-virginica"

global_counter = 0


def adaboost(S, y, H, k):
    """
    Parameters:
        S: Set of input points.
        y: Labels for the input points, taking values in {-1, 1}.
        H: Set of T weak classifiers, Each weak classifier is a function that takes an input x and returns {-1, 1}
        k: Number of iterations.

    Returns:
        classifiers_weights_list: list of weights for the classifiers
        most_important_classifiers: list of the most important weak classifiers
    """

    global global_counter

    num_of_points_in_S = len(S)
    D = np.ones(num_of_points_in_S) / num_of_points_in_S  # Initialize point weights

    classifiers_weights_list = []  # To store weights for multiple classifiers
    most_important_classifiers = []  # To store the chosen weak classifiers

    for t in range(k):
        # global_counter += 1
        # print(f"Global iteration: {global_counter}, local t: {t}")

        # List of weighted error for each weak classifier
        weighted_errors = []

        for h in H:
            try:
                predictions = np.array([h(x) for x in S])
                error = np.sum(D * (predictions != y))
                weighted_errors.append(error)
            except Exception as e:
                print(f"Error in classifier: {h}, error: {e}")
                raise

        # Select the classifier with minimum weighted error
        best_classifier_index = np.argmin(weighted_errors)
        best_classifier = H[best_classifier_index]
        min_error = weighted_errors[best_classifier_index]

        # Set classifier weight alpha_t based on its error
        alpha_t = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))  # add 1e-10 to avoid divide by 0
        classifiers_weights_list.append(alpha_t)
        most_important_classifiers.append(best_classifier)

        # Update the point weights
        predictions = np.array([best_classifier(x) for x in S])
        D = D * np.exp(-alpha_t * predictions * y)
        D /= np.sum(D)  # Normalize the weights

    return classifiers_weights_list, most_important_classifiers


def line_classifier_logic(p1, p2, x):
    # Cross product of the vectors p1p2 and p1x determines
    # the relative position of x with respect to the line
    return 1 if (p2[0] - p1[0]) * (x[1] - p1[1]) - (p2[1] - p1[1]) * (x[0] - p1[0]) > 0 else -1


def line_classifier(p1, p2):
    # Creates a classifier based on a line defined by two points p1 and p2
    return lambda x: line_classifier_logic(p1, p2, x)


def hypothesis_lines(S):
    # Generate all possible lines (hypotheses) from pairs of points in S

    num_of_points_in_S = len(S)
    hypotheses_set = []

    for i in range(num_of_points_in_S):
        for j in range(i + 1, num_of_points_in_S):
            try:
                hypotheses_set.append(line_classifier(S[i], S[j]))
            except ValueError as e:
                print(f"Skipping invalid line: {e}")

    return hypotheses_set


def compute_errors(S, y, classifiers_weights_list, classifiers_list):
    # Compute empirical and true errors for each stage of AdaBoost.

    num_of_points_in_S = len(S)
    F = np.zeros(num_of_points_in_S)  # F(x) in Final decision function
    empirical_errors_list = []

    for k in range(len(classifiers_list)):

        # Update F by adding the weighted prediction of the t-th classifier for all points in S
        F += classifiers_weights_list[k] * np.array([classifiers_list[k](x) for x in S])
        H_k = np.sign(F)
        empirical_errors_list.append(np.mean(H_k != y))  # add the error for the k-th stage to the list

    return empirical_errors_list


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


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # Load data from iris dataset
    data, labels = load_data("iris.txt", label2, label3)

    avg_empirical_errors = np.zeros(8)
    avg_true_errors = np.zeros(8)

    for _ in range(100):
        # Split the data randomly into train(S) and test(T) sets
        indexes = np.arange(len(data))
        np.random.shuffle(indexes)
        train_indexes = indexes[:len(data) // 2]
        test_indexes = indexes[len(data) // 2:]

        S_train = data[train_indexes]
        y_train = labels[train_indexes]
        T_test = data[test_indexes]
        y_test = labels[test_indexes]

        # Generate hypothesis set
        H = hypothesis_lines(S_train)

        # Run AdaBoost
        alpha, selected_classifiers = adaboost(S_train, y_train, H, 8)

        # Compute empirical and true errors
        empirical_errors = compute_errors(S_train, y_train, alpha, selected_classifiers)
        true_errors = compute_errors(T_test, y_test, alpha, selected_classifiers)

        avg_empirical_errors += np.array(empirical_errors)
        avg_true_errors += np.array(true_errors)

    # average of empirical and true errors
    avg_empirical_errors /= 100
    avg_true_errors /= 100

    print("Average empirical errors:", avg_empirical_errors)
    print("Average true errors:", avg_true_errors)
