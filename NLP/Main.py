import copy
import random
import sys

from nltk.corpus import brown
from sklearn.datasets import load_boston
from sklearn import linear_model
import m2cgen as m2c
import nltk
# nltk.download('brown')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as classifier
# from sklearn.linear_model import Perceptron as classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import random
from nltk import ngrams
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

def main():

    # for i in range(0, 30):
        # print(printList(brown.sents()[random.randint(0, len(brown.sents()))]))

    """
    words = []
    tags = []

    for tagged_word in brown.tagged_words():
        # Set tags to be Adjective, ADVerb, Determiner, Noun, Preposition, PROnoun, Verb, Conjunction
        word = tagged_word[0].lower()
        tag = tagged_word[1]

        # Remove negation (*)
        tag.replace("*", "")
        tag.replace("$", "Y")

        if word == "i" and (re.match(r"N[A-Z]*", tag) or re.match(r"P[A-Z]*", tag)):
            print(word)
            print(tag)

        # Remove hyphenations and prefixes
        if "+" in tag:
            if tag[-3:] == "+TO":
                tag = tag[0:-3]
        for i in range(0, 2):
            if "-" in tag:
                if tag[:3] == "FW-":
                    tag = tag[3:]
                elif tag[-3:] == "-HC" or tag[-3:] == "-TL" or tag[-3:] == "-NC":
                    tag = tag[0:-3]
        if "+" in tag:
            if tag[-3:] == "+TO":
                tag = tag[0:-3]

        # Pre-qualifier, article, post-determiner --> Preposition
        # A[A-Z]* (ABL, ABN, ABX, AP, AT) --> P
        if re.match(r"A[A-Z]*", tag):
            words.append(word)
            tags.append("P")
        # To be --> Verb
        # B[A-Z]* (BE, BED, BEDZ, BEG, BEM, BEN, BER, BBB) --> V
        elif re.match(r"B[A-Z]*", tag):
            words.append(word)
            tags.append("V")
        # Coordinating conjunction, subordinating conjunction --> Conjunction
        # CC, CS --> C
        elif tag == "CC" or tag == "CS":
            words.append(word)
            tags.append("C")
        # Cardinal numbers --> Adjective
        # CD --> A
        elif tag == "CD":
            words.append(word)
            tags.append("A")
        # To do --> Verb
        # DO[A-Z]* (DO, DOD, DOZ) --> V
        elif re.match(r"DO[A-Z]*", tag):
            words.append(word)
            tags.append("V")
        # Singular determiner, plural determiner --> Determiner
        # DT[A-Z]* (DT, DTI, DTS, DTX) --> D
        elif re.match(r"DT[A-Z]*", tag):
            words.append(word)
            tags.append("D")
        # Existential "there" --> Adjective
        # EX --> A
        elif tag == "EX":
            words.append(word)
            tags.append("A")
        # To have --> V
        # H[A-Z]* (HL, HV, HVD, HVG, HVN, HVZ) --> V
        elif re.match(r"H[A-Z]*", tag):
            words.append(word)
            tags.append("V")
        # Preposition --> Preposition
        # IN --> P
        elif tag == "IN":
            words.append(word)
            tags.append("P")
        # Adjective, comparative adjective, semantic adjective, morphological adjective --> Adjective
        # JJ[A-Z]* (JJ, JJR, JJS, JJT) --> A
        elif re.match(r"JJ[A-Z]*", tag):
            words.append(word)
            tags.append("A")
        # Modal auxiliary --> Verb
        # MD --> V
        elif tag == "MD":
            words.append(word)
            tags.append("V")
        # Possessive noun, plural noun, proper noun, singular noun, adverbial noun --> Noun
        # N[A-Z]* (NC, NN, NNY, NNS, NNSY, NP, NPY, NPS, NPSY, NR, NRS) --> N
        elif re.match(r"N[A-Z]*", tag):
            words.append(word)
            tags.append("N")
        # Ordinal numeral --> Adjective
        elif tag == "OD":
            words.append(word)
            tags.append("A")
        # Nominal pronoun, Possessive pronoun, personal pronoun, reflexive pronoun, objective pronoun, nominative
        # pronoun --> PROnoun
        # P[A-Z]* (PN, PN$, PPY, PPYY, PPL, PPLS, PPO, PPS, PPSS) --> PRO
        elif re.match(r"P[A-Z]*", tag):
            words.append(word)
            tags.append("PRO")
        # Qualifier --> ADVerb
        # QL --> ADV
        elif tag == "QL":
            words.append(word)
            tags.append("ADV")
        # Adverb, comparative, adverb, superlative adverb, nominal adverb, particle --> ADVerb
        # R[A-Z]* (RB, RBR, RBT, RBN, RP) --> ADV
        elif re.match(r"R[A-Z]*", tag):
            words.append(word)
            tags.append("ADV")
        # Verb in tense, verb --> Verb
        # VB[A-Z]* (VB, VBD, VBG, VBN, VBP, VBZ) --> V
        elif re.match(r"VB[A-Z]*", tag):
            words.append(word)
            tags.append("V")
        # Wh- determiner --> Determiner
        # WDT --> D
        elif tag == "WDT":
            words.append(word)
            tags.append("D")
        # Wh- pronoun --> PROnoun
        # WP[A-Z]* --> PRO
        elif re.match(r"WP[A-Z]*", tag):
            words.append(word)
            tags.append("PRO")
        # Wh- adverb --> ADVerb
        # WRB --> ADV
        elif tag == "WRB":
            words.append(word)
            tags.append("ADV")

    # print(words)
    # print(tags)

    d = {"words": words, "tags": tags}
    df = pd.DataFrame(data=d)

    df.to_csv("brown_relabelled.csv", index=False)

    """
    
    df = pd.read_csv("brown_relabelled.csv")

    count_vect = CountVectorizer()

    # df_rest, df = train_test_split(df, test_size=0.1)

    train, test = train_test_split(df, test_size=0.1)

    X_train, X_test, y_train, y_test = df.words, test.words, df.tags, test.tags

    # input = "i"
    # output = "PRO"
    # X_test = pd.core.series.Series(input.split(" "))
    # y_test = pd.core.series.Series(output.split(" "))

    X_train_counts = count_vect.fit_transform(X_train.values.astype('U'))
    X_test_counts = count_vect.transform(X_test.values.astype('U'))

    clf = classifier()
    # clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_counts, y_train)
    y_predict = clf.predict(X_test_counts)

    # occ, freq = letter_training(X_train, y_train, clf.classes_)
    # occ.to_csv("letter_occurrence", index=False)
    # freq.to_csv("letter_frequency.csv", index=False)

    # occ = pd.read_csv("letter_occurrence")
    # freq = pd.read_csv("letter_frequency.csv")
    # occ = pd.read_csv("occurrence.csv")
    # freq = pd.read_csv("frequency.csv")
    # y_predict = []
    # for word in X_test:
    #     # y_predict.append(letter_predict(word, occ, freq, clf.classes_))
    #     y_predict.append(word_predict(word, occ, freq, clf.classes_))
    # y_predict = np.array(y_predict)

    print(classification_report(y_test, y_predict))



def word_predict(word, occ, freq, classes):
    # Calculate prior probabilities
    prior = np.zeros(len(classes))
    for i in range(0, len(classes)):
        prior[i] = sum(freq[classes[i]])
    amount_of_items = sum(prior)
    for i in range(0, len(prior)):
        prior[i] = prior[i] / amount_of_items

    # prior = np.ones(len(classes))

    if not word in list(occ["WORD"]):
        prior  = prior *  (1 / len(classes))
    else:
        index = list(occ["WORD"]).index(word)
        for i in range(0, len(classes)):
            prior[i] = prior[i] * occ[classes[i]][index]

    return classes[argmax_class(prior)]

def letter_predict(word, occ, freq, classes):

    # Calculate prior probabilities
    prior = np.zeros(len(classes))
    for i in range(0, len(classes)):
        prior[i] = sum(freq[classes[i]])
    amount_of_items = sum(prior)
    for i in range(0, len(prior)):
        prior[i] = prior[i] / amount_of_items

    # prior = np.ones(len(classes))

    for character in word:
        if not character in list(occ["CHARACTERS"]):
            prior = prior * (1 / len(classes))
        else:
            index = list(occ["CHARACTERS"]).index(character)
            for i in range(0, len(classes)):
                prior[i] = prior[i] * occ[classes[i]][index]
    """
    character = word[0]
    if not character in list(occ["CHARACTERS"]):
        prior = prior * (1 / len(classes))
    else:
        index = list(occ["CHARACTERS"]).index(character)
        for i in range(0, len(classes)):
            prior[i] = prior[i] * occ[classes[i]][index]
    character = word[-1]
    if not character in list(occ["CHARACTERS"]):
        prior = prior * (1 / len(classes))
    else:
        index = list(occ["CHARACTERS"]).index(character)
        for i in range(0, len(classes)):
            prior[i] = prior[i] * occ[classes[i]][index]
    """



    # print(word)
    # print(classes[best_class])
    # print()
    return classes[argmax_class(prior)]



def unigram_training(X, y, classes):
    # print("je moeder")

    X = list(X)
    y = list(y)

    d1 = {}
    d1["A"] = {}
    d1["B"] = {}
    # print(d1)
    d1["A"]["the"] = 1
    if "the" in d1["A"]:
        d1["A"]["the"] += 1
    else:
        d1["A"]["the"] = 1
    # print(d1)

    tokens_per_classes = {}
    for c in classes:
        tokens_per_classes[c] = {}

    vocabulary = []

    # print(tokens_per_classes)

    for i in range(0, len(X)):
        if not X[i] in vocabulary:
            vocabulary.append(X[i])
        if X[i] in tokens_per_classes[y[i]]:
            tokens_per_classes[y[i]][X[i]] += 1
        else:
            tokens_per_classes[y[i]][X[i]] = 1

    # print(tokens_per_classes)
    # print(vocabulary)

    CLASSES = {}
    TOTAL = []

    for c in classes:
        CLASSES[c] = {c: []}

    # print(CLASSES)

    for word in vocabulary:
        totalCount = 0
        for c in classes:
            if word in tokens_per_classes[c]:
                CLASSES[c][c].append(tokens_per_classes[c][word])
                totalCount += tokens_per_classes[c][word]
            else:
                CLASSES[c][c].append(0)
        TOTAL.append(totalCount)

    LISTS = {}
    LISTS["WORD"] = vocabulary
    for c in classes:
        LISTS[c] = CLASSES[c][c]
    LISTS["TOTAL"] = TOTAL

    LISTS_OCCURENCE = copy.deepcopy(LISTS)

    for c in classes:
        for i in range(0, len(LISTS_OCCURENCE["TOTAL"])):
            LISTS_OCCURENCE[c][i] = LISTS_OCCURENCE[c][i] / LISTS_OCCURENCE["TOTAL"][i]

    trained_df_frequency = pd.DataFrame(data=LISTS)
    trained_df_occurence = pd.DataFrame(data=LISTS_OCCURENCE)
    # print(trained_df_frequency)
    # print((trained_df_occurence))

    return trained_df_occurence, trained_df_frequency
    
def letter_training(X, y, classes):
    print("je moeder")

    X = list(X)
    y = list(y)

    d1 = {}
    d1["A"] = {}
    d1["B"] = {}
    # print(d1)
    d1["A"]["the"] = 1
    if "the" in d1["A"]:
        d1["A"]["the"] += 1
    else:
        d1["A"]["the"] = 1
    # print(d1)

    tokens_per_classes = {}
    for c in classes:
        tokens_per_classes[c] = {}

    vocabulary = []

    print(tokens_per_classes)

    for i in range(0, len(X)):
        for l in str(X[i]):
            if not l in vocabulary:
                vocabulary.append(l)
            if l in tokens_per_classes[y[i]]:
                tokens_per_classes[y[i]][l] += 1
            else:
                tokens_per_classes[y[i]][l] = 1

    # print(tokens_per_classes)
    print(vocabulary)

    CLASSES = {}
    TOTAL = []

    for c in classes:
        CLASSES[c] = {c: []}

    print(CLASSES)

    for character in vocabulary:
        totalCount = 0
        for c in classes:
            if character in tokens_per_classes[c]:
                CLASSES[c][c].append(tokens_per_classes[c][character])
                totalCount += tokens_per_classes[c][character]
            else:
                CLASSES[c][c].append(0)
        TOTAL.append(totalCount)

    LISTS = {}
    LISTS["CHARACTERS"] = vocabulary
    for c in classes:
        LISTS[c] = CLASSES[c][c]
    LISTS["TOTAL"] = TOTAL

    LISTS_OCCURENCE = copy.deepcopy(LISTS)

    for c in classes:
        for i in range(0, len(LISTS_OCCURENCE["TOTAL"])):
            LISTS_OCCURENCE[c][i] = LISTS_OCCURENCE[c][i] / LISTS_OCCURENCE["TOTAL"][i]

    trained_df_frequency = pd.DataFrame(data=LISTS)
    trained_df_occurence = pd.DataFrame(data=LISTS_OCCURENCE)
    # print(trained_df_frequency)
    # print((trained_df_occurence))
    return trained_df_occurence, trained_df_frequency


def argmax_class(probs):
    best_prob = 0
    best_class = 0
    for i in range(0, len(probs)):
        if probs[i] > best_prob:
            best_prob = probs[i]
            best_class = i
    return best_class

def argmax_class_dict(probs):
    best_prob = 0
    best_class = ""
    for key in dict(probs).keys():
        if probs[key] > best_prob:
            best_prob = probs[key]
            best_class = key
    return best_class

def random_a_dict_and_sample_it( a_dictionary , a_number ):
    _ = {}
    for k1 in random.sample( list( a_dictionary.keys() ) , a_number ):
        _[ k1 ] = a_dictionary[ k1 ]
    return _


if __name__ == "__main__":
    main()