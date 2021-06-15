import random
import re
from collections import defaultdict

import pandas as pd
from nltk import trigrams, bigrams


def main():
    file = open("sentences_corrected.txt", "r+")
    direct_read = file.readlines()
    file.close()

    sentences = []
    for s in direct_read:
        if re.match(r"[0-9].*", s):
            sentences.append(s)
    random.shuffle(sentences)

    file = open("shuffled_sentences.txt", "w+")
    file.writelines(sentences)
    file.close()

    """
    df = pd.read_csv("brown_relabelled.csv")
    BIGRAM = []
    TRIGRAM = []
    BI_TRI_GRAM = []
    BI_TRI_GRAM_WEIGHTED = []
    for i in range(0, 10):
        print(i)
        text = printList(bigramPredictor("USE_RANDOM", 2, df, 1))
        BIGRAM.append(printList(bigramPredictor(text, 2, df, 30)))
        text = printList(bigramPredictor("USE_RANDOM", 2, df, 1))
        TRIGRAM.append(printList(trigramPredictor(text, 2, df, 30)))
        text = printList(bigramPredictor("USE_RANDOM", 2, df, 1))
        BI_TRI_GRAM.append(printList(bi_tri_gramPredictor(text, 2, df, 30, False)))
        text = printList(bigramPredictor("USE_RANDOM", 2, df, 1))
        BI_TRI_GRAM_WEIGHTED.append(printList(bi_tri_gramPredictor(text, 2, df, 30, True)))

    master_list = []
    master_list.append("BIGRAM\n")
    for s in BIGRAM:
        master_list.append(str(s) + "\n")
    master_list.append("\n")
    master_list.append("TRIGRAM\n")
    for s in TRIGRAM:
        master_list.append(str(s) + "\n")
    master_list.append("\n")
    master_list.append("BI_TRI_GRAM\n")
    for s in BI_TRI_GRAM:
        master_list.append(str(s) + "\n")
    master_list.append("\n")
    master_list.append("BI_TRI_GRAM_WEIGHTED\n")
    for s in BI_TRI_GRAM_WEIGHTED:
        master_list.append(str(s) + "\n")
    master_list.append("\n")

    file = open("sentences.txt", "w+")
    file.writelines(master_list)
    file.close()
    """


def bi_tri_gramPredictor(input, mode, df, max, use_weights):
    # Create a placeholder for model
    bi_model = defaultdict(lambda: defaultdict(lambda: 0))
    tri_model = defaultdict(lambda: defaultdict(lambda: 0))

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
            bi_model[(df[f][w1],)][df[t][w2]] += 1
    for w1, w2, w3 in trigrams(df.index, pad_right=True, pad_left=True):
        if (type(w1) == type(int("0")) and type(w2) == type(int("0")) and type(w3) == type(int("0"))):
            tri_model[(df[f][w1], df[f][w2])][df[t][w3]] += 1

    # Let's transform the counts to probabilities
    for w1 in bi_model:
        total_count = float(sum(bi_model[w1].values()))
        for w2 in bi_model[w1]:
            bi_model[w1][w2] /= total_count
    for w1_w2 in tri_model:
        total_count = float(sum(tri_model[w1_w2].values()))
        for w3 in tri_model[w1_w2]:
            tri_model[w1_w2][w3] /= total_count

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


def trigramPredictor(input, mode, df, max):
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

    # starting words
    text = input.split(" ")
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


def bigramPredictor(input, mode, df, max):
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
