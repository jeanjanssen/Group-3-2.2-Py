import numpy as np


class DecisionStump:

    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_col = X[:, self.feature_index]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_col < self.threshold] = -1
        else:
            predictions[X_col > self.threshold] = -1
        return predictions


class Adaboost:

    def __init__(self, n_classifiers=5):
        self.n_classifiers = n_classifiers

    # the training of the adaboost
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize the weights
        w = np.full(n_samples, (1 / n_samples))
        # here is where the training of the classifiers will take place. TODO: Add Haar classifiers

        self.classifiers = []
        for _ in range(self.n_classifiers):
            # This search method is greedy at the moment
            classifier = DecisionStump()

            min_error = float('inf')
            for feature_i in range(n_features):
                X_col = X[:, feature_i]
                thresholds = np.unique(X_col)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_col < self.threshold] = -1

                    missclassified = w[y != predictions]
                    error = sum(missclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        classifier.polarity = p
                        classifier.threshold = threshold
                        classifier.feature_index = feature_i

        epsilon = 1e-10
        classifier.alpha = 0.5 * np.log((1-error)/(error+epsilon))

        predictions = classifier.predict(X)

        w *= np.exp(-classifier.alpha*y*predictions)
        w /= np.sum(w)

        self.classifiers.append(classifier)

    def predict(self, X):
        classifier_predictions = [classifier.alpha(X) for classifier in self.classifiers]
        y_predictions = np.sum(classifier_predictions, axis=0)
        y_predictions = np.sign(y_predictions)
        return y_predictions