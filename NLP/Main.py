from nltk.corpus import brown
from sklearn.datasets import load_boston
from sklearn import linear_model
import m2cgen as m2c
import nltk
# nltk.download('brown')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import re

def main():
    words = []
    tags = []

    for tagged_word in brown.tagged_words():
        # Set tags to be Adjective, ADVerb, Determiner, Noun, Preposition, PROnoun, Verb, Conjunction
        word = tagged_word[0].lower()
        tag = tagged_word[1]

        # Remove negation (*)
        tag.replace("*", "")
        tag.replace("$", "Y")

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
        # WP[A-Z]*[$]* --> PRO
        elif re.match(r"WP[A-Z]*[$]*", tag):
            words.append(word)
            tags.append("PRO")
        # Wh- adverb --> ADVerb
        # WRB --> ADV
        elif tag == "WRB":
            words.append(word)
            tags.append("ADV")

    # print(words)
    # print(tags)

    count_vect = CountVectorizer()

    d = {"words": words, "tags": tags}
    df = pd.DataFrame(data=d)

    train, test = train_test_split(df, test_size=0.03)

    X_train, X_test, y_train, y_test = train.words, test.words, train.tags, test.tags

    input = "i eat bread"
    X_test = pd.core.series.Series(input.split(" "))
    # y_test = pd.core.series.Series(["D", "N", "V", "A", "N"])

    X_train_counts = count_vect.fit_transform(X_train)
    X_test_counts = count_vect.transform(X_test)

    print(X_train_counts.shape)
    print(y_train.shape)
    print(X_test_counts.shape)
    print(y_test.shape)

    clf = MultinomialNB()
    # clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_counts, y_train)
    y_predict = clf.predict(X_test_counts)

    # print(classification_report(y_test, y_predict, zero_division=0))

    print(list(X_test))
    # print(list(y_test))
    print(list(y_predict))

    # code = m2c.export_to_java(clf)
    # print(code)

    """
    boston = load_boston()
    X, y = boston.data, boston.target

    estimator = linear_model.LinearRegression()
    estimator.fit(X, y)

    code = m2c.export_to_java(estimator)
    """

if __name__ == "__main__":
    main()