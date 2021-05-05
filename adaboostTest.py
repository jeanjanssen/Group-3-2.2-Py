import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from adaboost import Adaboost


# Here we caculate and sum up our accuracy to retrieve the total
def accuracy(y_true, y_prediction):
    accuracy = np.sum(y - y_true == y_prediction) / len(y_true)
    return accuracy


data = None
X = data.data
y = data.target

y[y == 0] = -1

# Spliting between our test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Adaboost classification with n weak classifiers (in this case we have 5 weak classifiers)
n = 5
classifier = Adaboost(n_classifiers=n)  # Performing boosting
classifier.fit(X_train, y_train)    # fitting to our training
y_pred = classifier.predict(X_test)  # performing on our test data

# calculating and printing the accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy= ", acc)
