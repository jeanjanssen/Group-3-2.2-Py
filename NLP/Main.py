from nltk.corpus import brown
from sklearn.datasets import load_boston
from sklearn import linear_model
import m2cgen as m2c
import nltk
# nltk.download('brown')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer


def main():
    words = []
    tags = []

    for tagged_word in brown.tagged_words():
        # Set tags to be
        words.append(tagged_word[0])
        tags.append(tagged_word[1])

    # print(words)
    # print(tags)

    count_vect = CountVectorizer()

    d = {"words": words, "tags": tags}
    df = pd.DataFrame(data=d)

    train, test = train_test_split(df, test_size=0.2)

    X_train_counts = count_vect.fit_transform(train.words)


    print(train.shape)
    print(test.shape)

    X_train, X_test, y_train, y_test = train.words, train.tags, test.words, test.tags

    clf = MultinomialNB().fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    print(classification_report(y_test, y_predict))

    """
    boston = load_boston()
    X, y = boston.data, boston.target

    estimator = linear_model.LinearRegression()
    estimator.fit(X, y)

    code = m2c.export_to_java(estimator)
    """

if __name__ == "__main__":
    main()