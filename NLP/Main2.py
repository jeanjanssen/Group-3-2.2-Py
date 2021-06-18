import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk import trigrams, bigrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from NLP.Main import word_predict
from NLP.Main import argmax_class_dict


def main():
    df = pd.read_csv("brown_relabelled.csv")

    X_train, X_test, y_train, y_test = df.words, df.words[20000:120000], df.tags, df.tags[20000:120000]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train.values.astype('U'))
    X_test_counts = count_vect.transform(X_test.values.astype('U'))

    clf_NB = MultinomialNB()
    clf_NB.fit(X_train_counts, y_train)
    y_predict_NB = clf_NB.predict(X_test_counts)

    occ = pd.read_csv("occurrence.csv")
    freq = pd.read_csv("frequency.csv")
    y_predict_selfNB = []
    y_predict_NG = []
    y_predict_NGNB = []
    tri_model = createTriModel(1, df)
    for i in X_test.index:
        y_predict_selfNB.append(word_predict(X_test[i], occ, freq, clf_NB.classes_))
        y_predict_NG.append(trigramPredictor(df.words[i-2] + " " + df.words[i-2], df, 1, tri_model, 1)[-1])
        y_predict_NGNB.append(NGNB(df.words[i-2] + " " + df.words[i-2], df, 1, tri_model, 1, X_test[i], occ, freq, clf_NB.classes_))
        if i % 5000 == 0:
            print(str((i - 20000) / 1000) + "%")
    y_predict_selfNB = np.array(y_predict_selfNB)
    y_predict_NG = np.array(y_predict_NG)
    y_predict_NGNB = np.array(y_predict_NGNB)

    CR_NB = classification_report(y_test, y_predict_NB)
    CR_selfNB = classification_report(y_test, y_predict_selfNB)
    CR_NG = classification_report(y_test, y_predict_NG)
    CR_NGNB = classification_report(y_test, y_predict_NGNB)

    CR_dict = {
        "CR_NB": CR_NB,
        "CR_selfNB": CR_selfNB,
        "CR_NG": CR_NG,
        "CR_NGNB": CR_NGNB
    }

    for key in CR_dict.keys():
        print(key)
        print(CR_dict[key])
        print()

def NGNB(input, df, max, model, mode, word, occ, freq, classes):
    # Calculate prior probabilities
    prior = np.zeros(len(classes))
    for i in range(0, len(classes)):
        prior[i] = sum(freq[classes[i]])
    amount_of_items = sum(prior)
    for i in range(0, len(prior)):
        prior[i] = prior[i] / amount_of_items

    probs_final = {}

    if not word in list(occ["WORD"]):
        prior = prior * (1 / len(classes))
    else:
        index = list(occ["WORD"]).index(word)
        for i in range(0, len(classes)):
            prior[i] = prior[i] * occ[classes[i]][index]
            probs_final[classes[i]] = prior[i]

    # Types: 0 - tag from tags, 1 - tag from words, 2 - word from words, 4 word from tags
    f = "words"
    t = "words"
    if mode == 0:
        f = "tags"
        t = "tags"
    elif mode == 1:
        t = "tags"
    elif mode == 4:
        f = "tags"

    # starting words
    text = input.split(" ")
    original_text = text.copy()
    sentence_finished = False

    current = 1
    while not sentence_finished:
        # select a random probability threshold
        r = random.random() + 10000
        best = .0
        current_prob = .0
        accumulator = .0
        best_word = ""

        text_copy = text.copy()
        while not tuple(text[-2:]) in model:
            text_copy[-2] = df[f][random.randint(0, len(df[f]) - 1)]
            if not tuple(text_copy[-2:]) in model:
                text_copy = text.copy()
                text_copy[-1] = df[f][random.randint(0, len(df[f]) - 1)]
                if not tuple(text_copy[-2:]) in model:
                    text_copy[-2] = df[f][random.randint(0, len(df[f]) - 1)]
            text = text_copy.copy()

        for word in model[tuple(text[-2:])].keys():
            # accumulator += model[tuple(text[-2:])][word]
            current_prob = model[tuple(text[-2:])][word]
            if word in probs_final:
                probs_final[word] += current_prob
                probs_final[word] /= 2

        if text[-2:] == [None, None] or current >= max:
            sentence_finished = True
        current += 1

    return argmax_class_dict(probs_final)

def createTriModel(mode, df):
    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Types: 0 - tag from tags, 1 - tag from words, 2 - word from words, 4 word from tags
    f = "words"
    t = "words"
    if mode == 0:
        f = "tags"
        t = "tags"
    elif mode == 1:
        t = "tags"
    elif mode == 4:
        f = "tags"

    # Count frequency of co-occurance
    for w1, w2, w3 in trigrams(df.index, pad_right=True, pad_left=True):
        if (type(w1) == type(int("0")) and type(w2) == type(int("0")) and type(w3) == type(int("0"))):
            model[(df[f][w1], df[f][w2])][df[t][w3]] += 1

    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    return model

def createBiModel(mode, df):
    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Types: 0 - tag from tags, 1 - tag from words, 2 - word from words, 4 word from tags
    f = "words"
    t = "words"
    if mode == 0:
        f = "tags"
        t = "tags"
    elif mode == 1:
        t = "tags"
    elif mode == 4:
        f = "tags"

    # Count frequency of co-occurance
    for w1, w2 in bigrams(df.index, pad_right=True, pad_left=True):
        if type(w1) == type(int("0")) and type(w2) == type(int("0")):
            model[(df[f][w1],)][df[t][w2]] += 1

    # Let's transform the counts to probabilities
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count
    return model

def bi_tri_gramPredictor(input, df, max, use_weights, bi_model, tri_model, mode):
    # Types: 0 - tag from tags, 1 - tag from words, 2 - word from words, 4 word from tags
    f = "words"
    t = "words"
    if mode == 0:
        f = "tags"
        t = "tags"
    elif mode == 1:
        t = "tags"
    elif mode == 4:
        f = "tags"

    # starting words
    text = input.split(" ")
    original_text = text.copy()
    sentence_finished = False

    current = 1
    while not sentence_finished:
        # select a random probability threshold
        r = random.random() * 4
        best = .0
        accumulator = .0
        best_word = ""

        text_copy = text.copy()
        while not tuple(text[-2:]) in tri_model:
            text_copy[-2] = df[f][random.randint(0, len(df[f]) - 1)]
            if not tuple(text_copy[-2:]) in tri_model:
                text_copy = text.copy()
                text_copy[-1] = df[f][random.randint(0, len(df[f]) - 1)]
                if not tuple(text_copy[-2:]) in tri_model:
                    text_copy[-2] = df[f][random.randint(0, len(df[f]) - 1)]
            text = text_copy.copy()
        for word in random.sample(list(tri_model[tuple(text[-2:])].keys()), len(tri_model[tuple(text[-2:])].keys())):
            weight = 1
            if use_weights:
                weight = random.randint(1, 5)
            # accumulator += (weight * tri_model[tuple(text[-2:])][word] + bi_model[tuple(text[-1:])][word]) / (1 + weight)
            current_prob = (weight * tri_model[tuple(text[-2:])][word] + bi_model[tuple(text[-1:])][word]) / (1 + weight)
            # select words that are above the probability threshold
            if current_prob > best or abs(current_prob - best) < 0.1:
                best_word = word
                best = current_prob
            if accumulator > r:
                break
        text.append(best_word)
        original_text.append(best_word)

        if text[-2:] == [None, None] or current >= max:
            sentence_finished = True
        current += 1
    return original_text


def trigramPredictor(input, df, max, model, mode):
    # Types: 0 - tag from tags, 1 - tag from words, 2 - word from words, 4 word from tags
    f = "words"
    t = "words"
    if mode == 0:
        f = "tags"
        t = "tags"
    elif mode == 1:
        t = "tags"
    elif mode == 4:
        f = "tags"

    # starting words
    text = input.split(" ")
    original_text = text.copy()
    sentence_finished = False

    current = 1
    while not sentence_finished:
        # select a random probability threshold
        r = random.random() + 10000
        best = .0
        current_prob = .0
        accumulator = .0
        best_word = ""

        text_copy = text.copy()
        while not tuple(text[-2:]) in model:
            text_copy[-2] = df[f][random.randint(0, len(df[f]) - 1)]
            if not tuple(text_copy[-2:]) in model:
                text_copy = text.copy()
                text_copy[-1] = df[f][random.randint(0, len(df[f]) - 1)]
                if not tuple(text_copy[-2:]) in model:
                    text_copy[-2] = df[f][random.randint(0, len(df[f]) - 1)]
            text = text_copy.copy()

        for word in model[tuple(text[-2:])].keys():
            # accumulator += model[tuple(text[-2:])][word]
            current_prob = model[tuple(text[-2:])][word]
            # select words that are above the probability threshold
            if current_prob > best:
                best_word = word
                best = current_prob
            if accumulator > r:
                break
        text.append(best_word)
        original_text.append(best_word)

        if text[-2:] == [None, None] or current >= max:
            sentence_finished = True
        current += 1
    return original_text


def bigramPredictor(input, df, max, model, mode):
    # Types: 0 - tag from tags, 1 - tag from words, 2 - word from words, 4 word from tags
    f = "words"
    t = "words"
    if mode == 0:
        f = "tags"
        t = "tags"
    elif mode == 1:
        t = "tags"
    elif mode == 4:
        f = "tags"

    # starting words
    text = input.split(" ")
    if input == "USE_RANDOM":
        text[-1] = df[f][random.randint(0, len(df[f]) - 1)]
    original_text = text.copy()
    sentence_finished = False

    current = 1
    while not sentence_finished:
        # select a random probability threshold
        r = random.random() * 4
        best = .0
        current_prob = .0
        accumulator = .0
        best_word = ""

        while not tuple(text[-1:]) in model:
            text[-1] = df[f][random.randint(0, len(df[f]) - 1)]

        for word in model[tuple(text[-1:])].keys():
            # accumulator += model[tuple(text[-1:])][word]
            current_prob = model[tuple(text[-1:])][word]
            # select words that are above the probability threshold
            if current_prob > best:
                best_word = word
                best = current_prob
            if accumulator > r:
                break
        text.append(best_word)
        original_text.append(best_word)

        if text[-1:] == [None] or current >= max:
            sentence_finished = True
        current += 1
    return original_text


def printList(l):
    T = ' '.join([str(t) for t in l if t])
    # print(T)
    return T


if __name__ == "__main__":
    main()
