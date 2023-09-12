import sys

import pandas as pd
from decision_tree import DecisionTreeClassifier as dct
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def score(y, y_hat):
    """

    :param y: real values
    :param y_hat: predicted values
    :return: hits in percent
    """
    return float(y[y == y_hat].size) / float(y.size) * 100.


data = pd.read_csv("data/Surgical-deepnet.csv")
data.dropna(inplace=True)

y = data['complication'].copy()
X = data.drop('complication', axis=1).copy()

X_train_raw, X_test_raw, y_train_raw, y_test_raw = \
    train_test_split(X, y, stratify=y, train_size=0.8, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = X_train_raw.to_numpy(), X_test_raw.to_numpy(), y_train_raw.to_numpy(),\
                                   y_test_raw.to_numpy()
y_train.shape = (-1, 1)
y_test.shape = (-1, 1)

model = DecisionTreeClassifier(max_depth=5, criterion='entropy')
model.fit(X_train, y_train)
train_res = model.predict(X_train)
test_res = model.predict(X_test)

model = dct.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
train_res_dct = model.predict(X_train)
test_res_dct = model.predict(X_test)

original_stdout = sys.stdout
with open("output.txt", 'w') as output:
    sys.stdout = output
    pd.set_option('display.max_columns', None)
    print(data.head())

    print("My categorical classifier:")
    print("Train score = {}%".format(score(y_train, train_res_dct)))
    print("Test score = {}%".format(score(y_test, test_res_dct)))

    print("Sklearn classifier:")
    print("Train score = {}%".format(score(y_train, train_res.reshape(-1, 1))))
    print("Test score = {}%".format(score(y_test, test_res.reshape(-1, 1))))
    sys.stdout = original_stdout
