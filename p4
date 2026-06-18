import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# Load the dataset
iris = datasets.load_iris()
X = iris.data # Features
y = iris.target # Target labels
# Split the dataset into training and testing sets (80% training, 20%
testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Function to evaluate k-NN with different k values
def evaluate_knn(k_values, X_train, X_test, y_train, y_test,
weighted=False):
results = []
for k in k_values:
# Initialize k-NN classifier with or without distance-based
weights
if weighted:
knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
else:

knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
# Train the classifier
knn.fit(X_train, y_train)
# Make predictions
y_pred = knn.predict(X_test)
# Calculate accuracy and F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted') # Weighted
F1-score multi-class
results.append((k, accuracy, f1))
return results
# Test k-NN for different values of k for unweighted and weighted classes
k_values = [1, 3, 5]
# Unweighted k-NN
print("Unweighted k-NN:")
unweighted_results = evaluate_knn(k_values, X_train, X_test, y_train,
y_test, weighted=False)
for k, accuracy, f1 in unweighted_results:
print(f"k={k}, Accuracy={accuracy:.4f}, F1-score={f1:.4f}")
# Weighted k-NN
print("Weighted k-NN:")
weighted_results = evaluate_knn(k_values, X_train, X_test, y_train,
y_test, weighted=True)
for k, accuracy, f1 in weighted_results:
print(f"k={k}, Accuracy={accuracy:.4f}, F1-score={f1:.4f}")
# Visualize the results
weighted_accuracies=[]
unweighted_accuracies=[]
for k,accuracy, f1 in unweighted_results:
unweighted_accuracies.append(accuracy)


for k, accuracy, f1 in weighted_results:
weighted_accuracies.append(accuracy)
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, unweighted_accuracies, label='Unweighted k-NN',
marker='o')
plt.plot(k_values, weighted_accuracies, label='Weighted k-NN', marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Accuracy of k-NN with Different k Values')
plt.show()
