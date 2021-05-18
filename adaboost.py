'''
In machine learning, boosting is an ensemble technique used for trying to create a strong classifier from a set of weak
classifiers. We use this for predictive modelling problems. To iterate on boosting, we first build a model from the
training data, and then create a second model utilizing the errors from the first model to better correct it in the
second model. Models continue to be added until the training set is predicted perfectly, or we reach the limited
number of models that we will make. In addition, it should be noted that we do not want to overfit our model to the
training data.

History on adaboost, adaboost was the first really successful boosting algorithm developed for binary classification.
How do we learn an adaboost model from given data? We know that we use adaboosting to boost the performance of decision
trees on binary classification problems. Side note: it is best used on weak classifiers (how to define weak learners:
these learners achieve an accuracy just above random chance on a classification problem). The most suited case is to
use adaboost with decision trees with one level. This is because these trees are so short and only include one
decision for classification. They are often called decision stumps.

Adaboost ensemble: weak classifiers are added sequentially. They are trained using the weighted training data. After
you have the number of weak classifiers you want, you are left with a pool of weak learners each with a stage value.

Making predictions: prediction are made by calculating the weighted average of the weak classifiers. For each new
input instance, each weak learner calculates a predicted value as either +1.0 or -1.0.

Note: Some stumps get more say in the classification than others (they dont have equal importance).
Each stump is made bt taking the previous stump's mistakes into account (therefore, order of the stumps is important).

'''

import numpy as np


class DecisionStump:

    def __init__(self):
        self.polarity = 1  # Polarity is for indicating the direction of the inequality (taking one classification
        # over the other i.e. error over correctness and vice versa)
        self.feature_index = None  # Feature index indicates the feature we specify
        self.threshold = None  # Threshold is set for our accuracy in training
        self.alpha = None  # Alpha is the weight (amount of say) for each decision stump

    # The prediction method will take in our image. This will return the predictions for each decision stump.
    def predict(self, X):
        n_samples = X.shape[0]
        X_col = X[:, self.feature_index]  # We will look down by columns for on a specific feature index.

        predictions = np.ones(n_samples)  # Looking at our samples, we will convert them to 1's in our matrix.
        if self.polarity == 1:  # We will calculate our predictions given our polarity == 1
            predictions[X_col < self.threshold] = -1  # Our prediction samples will be 1 and -1 for anything below
            # the threshold
        else:
            predictions[X_col > self.threshold] = -1  # Taking the opposite, we classify everything above the threshold
            # as -1
        return predictions


class Adaboost:

    def __init__(self, n_classifiers=5):  # Here we define the number of weak classifiers we want to perform boosting on
        self.n_classifiers = n_classifiers
    
    x_coord = 1
    y_coord = 1
    w = 200
    h = 300
    feature = Haar()
    feature_score = 0
    im_list = feature.image_list
    for i in range(len(im_list)):
        feature_score = feature_score + feature.edge_vertical(x_coord, y_coord, w, h, i)
    final_score = feature_score / len(im_list)
    
    # the training of the adaboost
    def fit(self, X, y):
        n_samples, n_features = X.shape  # Defining our samples and features in order to fit below

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
                    predictions[X_col < self.threshold] = -1  # We set above and below the threshold depending on the
                    # polarity value

                    missclassified = w[y != predictions]
                    error = sum(missclassified)  # We get the total error on our misclassified instances

                    if error > 0.5:  # This is above the random classification, which is exactly what we want in
                        # order to add value to the final classification (weak classifier).
                        error = 1 - error
                        p = -1  # Do add value to the final decision, we can reverse the polarity and take the other
                        # side of the accuracy gain

                    if error < min_error:  # This checks the new calculated error. We will reset everything based on
                        # this. Therefore, we can use it for the next iteration.
                        min_error = error
                        classifier.polarity = p
                        classifier.threshold = threshold
                        classifier.feature_index = feature_i

        epsilon = 1e-10  # initializing epsilon which will be used in the below equation
        # This is the equation for the amount of say for each of the decision stumps has on the final classification
        classifier.alpha = 0.5 * np.log((1 - error) / (error + epsilon))

        predictions = classifier.predict(X)  # running the prediction

        w *= np.exp(-classifier.alpha * y * predictions)  # equation for calculating the weight
        w /= np.sum(w)  # summing the instance weights

        self.classifiers.append(classifier)

    # Here we perform predictions utilizing each decision stump
    def predict(self, X):
        classifier_predictions = [classifier.alpha(X) for classifier in self.classifiers]
        y_predictions = np.sum(classifier_predictions, axis=0)
        y_predictions = np.sign(y_predictions)
        return y_predictions
