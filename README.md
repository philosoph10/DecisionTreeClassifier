# Decision Tree Classifier from Scratch

A simple implementation of a decision tree classifier that accepts both categorical and continuous data, along with a discretizer for transforming continuous data into categories. This implementation is tested on the Surgical-Deepnet dataset and achieves 78% train accuracy and 77% test accuracy. For reference, sklearn's decision tree classifier obtains 83% accuracy on both train and test.

## Usage

### Discretizer

The `Discretizer` class is responsible for converting continuous data into categories using a K-Means clustering algorithm. You can use it as follows:

```python
from decision_tree import Discretizer

# Create a Discretizer with the desired number of categories
discretizer = Discretizer(num_categories=10)

# Fit the discretizer to your data
discretizer.fit(X)

# Transform your data into categories
X_transformed = discretizer.transform(X)
```

## Decision Tree Classifier

The `DecisionTreeClassifier` class performs binary classification using a decision tree algorithm. Here's how you can use it:

```python
from decision_tree import DecisionTreeClassifier

# Create a DecisionTreeClassifier with an optional maximum depth
clf = DecisionTreeClassifier(max_depth=-1)

# Fit the classifier to your training data
clf.fit(X_train, y_train)

# Make predictions on new data
y_pred = clf.predict(X_test)
```

## Dependencies

This project relies on the following dependencies:

- Python 3.x
- NumPy

## Acknowledgments

This implementation was developed as a learning exercise and may not be as optimized or feature-rich as established libraries. However, it provides a simple example of how a decision tree classifier and a discretizer can be built from scratch.

Happy coding!
