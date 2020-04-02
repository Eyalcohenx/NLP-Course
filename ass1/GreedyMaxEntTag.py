import re
import sys
from joblib import load
from collections import defaultdict, namedtuple
import numpy as np
from sklearn.metrics import precision_score, recall_score
import warnings

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", message="Recall is ill-defined")

input_file_name = sys.argv[1]
model_name = sys.argv[2]
feature_map_file = sys.argv[3]
out_file_name = sys.argv[4]
out_file = open(out_file_name, "w")


def check_availability(element, collection: iter):
    return element in collection


class HasNextIterator:
    def __init__(self, it):
        self._it = iter(it)
        self._next = None

    def __iter__(self):
        return self

    def has_next(self):
        if self._next:
            return True
        try:
            self._next = next(self._it)
            return True
        except StopIteration:
            return False

    def next(self):
        if self._next:
            ret = self._next
            self._next = None
            return ret
        elif self.has_next():
            return self.next()
        else:
            raise StopIteration()


def process_line(line):
    return line.split()


def read_file(file):
    with open(file, 'r') as file:
        return [process_line(line) for line in file.readlines()]


signature = namedtuple('Signature', ['pattern', 'name'])
signatures = [signature(re.compile(r'^\d+'), 'NUM'), \
              signature(re.compile(r'^.*\-+.*'), 'HYPHEN'), \
              signature(re.compile(r'^.*[A-Z].*'), 'CAPITAL'), \
              ]

test_sents = read_file(input_file_name)

clf = load(model_name)

feature_mapping = load(feature_map_file)
possible_tags, vec, le = feature_mapping


def check_if_rare_with_vectorizer(word):
    if check_availability("w_i=" + word, vec.get_feature_names()):
        return False
    return True

def word_to_features(tok, check_if_rare, word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1,
                     word_plus_1, word_plus_2):
    features = defaultdict()
    if check_if_rare(tok):
        for sign in signatures:
            if sign.pattern.match(tok):
                features[sign.name] = "True"
        try:
            prefix_1 = tok[0:1]
            features["prefix_1"] = prefix_1
        except EnvironmentError:
            pass
        try:
            prefix_2 = tok[0:2]
            features["prefix_2"] = prefix_2
        except EnvironmentError:
            pass
        try:
            prefix_3 = tok[0:3]
            features["prefix_3"] = prefix_3
        except EnvironmentError:
            pass
        try:
            prefix_4 = tok[0:4]
            features["prefix_4"] = prefix_4
        except EnvironmentError:
            pass
        try:
            suffix_1 = tok[-1:]
            features["suffix_1"] = suffix_1
        except EnvironmentError:
            pass
        try:
            suffix_2 = tok[-2:]
            features["suffix_2"] = suffix_2
        except EnvironmentError:
            pass
        try:
            suffix_3 = tok[-3:]
            features["suffix_3"] = suffix_3
        except EnvironmentError:
            pass
        try:
            suffix_4 = tok[-4:]
            features["suffix_4"] = suffix_4
        except EnvironmentError:
            pass
    else:
        features["w_i"] = tok

    features["tag-1"] = tag_minus_1
    features["tag-2-1"] = tag_minus_2_1
    features["word-1"] = word_minus_1
    features["word-2"] = word_minus_2
    features["word+1"] = word_plus_1
    features["word+2"] = word_plus_2
    return features

START = '<START>'
END = '<END>'

y_predicted = []


def greedy_predict(test_sents, out_file):
    for sent in test_sents:

        predictions_for_line = []
        it = HasNextIterator(sent)

        if it.has_next():
            it.next()

        if it.has_next():
            word_plus_1 = it.next()
        else:
            word_plus_1 = END

        if it.has_next():
            word_plus_2 = it.next()
        else:
            word_plus_2 = END

        word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = "START", START + "_" + START, START, "START"

        for tok in sent:

            features = word_to_features(tok, check_if_rare_with_vectorizer, word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1, word_plus_1, word_plus_2)

            # greedy tagger, we take the best tag at each point
            tag = clf.predict(vec.transform(features))[0]
            tag = le.inverse_transform([tag])[0]

            y_predicted.append(tag)
            predictions_for_line.append(tag)

            if it.has_next():
                word_plus_1 = it.next()
            else:
                word_plus_1 = END

            if it.has_next():
                word_plus_2 = it.next()
            else:
                word_plus_2 = END

            word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = word_minus_1, tag_minus_1 + "_" + tag, tag, tok

        for tok, tag in zip(sent, predictions_for_line):
            print(tok + "/" + tag, end=" ", file=out_file)
        print("", file=out_file)


greedy_predict(test_sents, out_file)


# TEST:

def process_line_y(line):
    temp = [re.search(r'(.*)/(.*$)', pair).groups() for pair in line.split()]
    return [t[1] for t in temp]


def read_file_y(file):
    with open(file, 'r') as file:
        return [process_line_y(line) for line in file.readlines()]

y_predicted = le.transform(y_predicted)
y_test = read_file_y('data/ass1-tagger-dev')
y_test = [val for sublist in y_test for val in sublist]
y_test = le.transform(y_test)

precision = precision_score(y_test, y_predicted, average="weighted")
recall = recall_score(y_test, y_predicted, average="weighted")
acc = np.mean(y_test == y_predicted)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", acc)
