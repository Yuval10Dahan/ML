import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# path = "C:\\Users\\yuval\\Desktop\\Code\\PyCharm Code Projects\\ML\\Ex2\\iris.txt"

# User input for labels
label1 = "Iris-setosa"
label2 = "Iris-versicolor"
label3 = "Iris-virginica"

number_of_mistakes = 0


def load_data(file_path, type1, type2):
    data = []  # store the data from the file
    labels = []  # store the class labels

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
                data.append([second_feature, third_feature])
                labels.append(1)  # Assign 1 for `type1`
            elif label == type2:
                data.append([second_feature, third_feature])
                labels.append(-1)  # Assign -1 for `type2`

    return np.array(data), np.array(labels)


# Perceptron training algorithm
def perceptron(data, labels):
    global number_of_mistakes  # Use the global variable to track mistakes
    number_of_mistakes = 0

    # Initialize the weight vector w
    w = np.zeros(data.shape[1])

    # t rounds
    while True:
        no_mistakes = True

        # Iterate over all points x_i
        for i, x_i in enumerate(data):
            true_label = labels[i]

            # If w_t * x_i > 0, guess 1; else, guess -1
            if np.dot(w, x_i) > 0:
                guess = 1
            else:
                guess = -1

            # If guess is wrong
            if guess != true_label:
                if true_label == 1:  # If x is really 1
                    w += x_i  # Update w_t+1 = w_t + x_i
                elif true_label == -1:  # If x is really -1
                    w -= x_i  # Update w_t+1 = w_t - x_i
                number_of_mistakes += 1
                no_mistakes = False
                break  # Exit round t

        # If no mistakes this round, exit algorithm
        if no_mistakes:
            break

    return w


# choose desired species
data, labels = load_data("iris.txt", label1, label2)
weights = perceptron(data, labels)
print("Final weight vector:", weights)
print("Number of mistakes:", number_of_mistakes)

# Fit SVM
svm = SVC(kernel='linear', C=1e10)  # Large C approximates hard margin SVM
svm.fit(data, labels)

# Get weights and margin
w = svm.coef_[0]
margin = 1 / np.linalg.norm(w)
print("True Maximum Margin:", margin)


# Plot the data and decision boundary
plt.figure(figsize=(8, 6))
for x, y in zip(data, labels):
    if y == 1:
        plt.scatter(x[0], x[1], color="orange", label=label1 if label1 not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(x[0], x[1], color="blue", label=label2 if label2 not in plt.gca().get_legend_handles_labels()[1] else "")

# Decision boundary
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
x_values = np.linspace(x_min, x_max, 100)
y_values = -(weights[0] * x_values) / weights[1]
plt.plot(x_values, y_values, color="red", label="Decision Boundary")

plt.xlabel("Feature 2")
plt.ylabel("Feature 3")
plt.legend()
plt.title(f"Perceptron Algorithm: {label1} vs {label2}")
plt.show()
