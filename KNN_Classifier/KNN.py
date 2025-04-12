import numpy as np

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

# Encoding  the string labels ('Apple', 'Banana', 'Orange').
# Examples:
# Labels:
# Apple → 0
# Banana → 1
# Orange → 2

labels = {'Apple': 0, 'Banana': 1, 'Orange': 2}
encoded_data = np.array([[item[0], item[1], item[2], labels[item[3]]] for item in data])

# Seperating data
X = encoded_data[:, :-1]  # First three columns of given data(Feautue Matrix)
y = encoded_data[:, -1]   # Last column of given data (Label Vector)


# Eucledean distance function between two points in N-dimensional space sqrt(sum((a-b)^2))
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # predicting the labels for the test set
        predictions = [self.predict_one(x) for x in X_test]
        return np.array(predictions)

    def predict_one(self, x):
        # Calculating distances to all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sorting the distances calculated above and getting K nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Calculating the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # COunting the no of occurrences of each label in the k nearest neighbor
        values, counts = np.unique(k_nearest_labels, return_counts=True)
        # calcualting the maximum count 
        max_index = max(enumerate(counts), key=lambda x: x[1])[0]
        # Most common label
        most_common_label = values[max_index]
        return most_common_label

test_data = np.array([
    [118, 6.2, 0],  # Banana
    [160, 7.3, 1],  # Apple
    [185, 7.7, 2]   # Orange
])

KNN = KNNClassifier(k=3)
KNN.fit(X, y)
predictions = KNN.predict(test_data)
print("Predictions: ", predictions)
print("Converted Predictions: ", [list(labels.keys())[list(labels.values()).index(pred)] for pred in predictions])
# The predicted output should be:
# Predictions: [1 0 2]
# Converted Predictions: ['Banana', 'Apple', 'Orange']
# The predictions should match the expected labels for the test data points.

# Evaluating the model with different value of k
K_parameters = [1, 3, 5]
for k in K_parameters:
    KNN = KNNClassifier(k=k)# creating new instance with k value
    KNN.fit(X, y) # fitting the model
    predictions = KNN.predict(test_data) # predicting the data
    print(f"Predictions for K=k: ", predictions) 
    print(f"Converted Predictions for K=k: ", [list(labels.keys())[list(labels.values()).index(pred)] for pred in predictions])
# The output will show the predicted labels for the test data points with different values of k.
# The expected output should be as follows:
# Predictions for k=1: [1 0 2]
# Converted Predictions for k=1: ['Banana', 'Apple', 'Orange']
# Predictions for k=3: [1 0 2]
# Converted Predictions for k=3: ['Banana', 'Apple', 'Orange']
# Predictions for k=5: [1 0 2]
# Converted Predictions for k=5: ['Banana', 'Apple', 'Orange']

# Bonus challenge:
# Implementing the accuracy checker function
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# NOrmalizing the data using Min-Max scaling
def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

# Basic training and testing split
# Split the data into training and testing sets (75% train, 25% test)
train_size = int(0.75 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]


# Using manhattan distance
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# Using Minkowski distance
def minkowski_distance(a, b, p=4):
    return np.sum(np.abs(a - b) ** p) ** (1/p)


