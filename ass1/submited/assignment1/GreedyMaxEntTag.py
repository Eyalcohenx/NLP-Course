#GreedyMaxEntTag
import re
import sys
from joblib import load
from collections import defaultdict, namedtuple
import numpy as np
from sklearn.metrics import precision_score, recall_score
from ExtractFeatures import signatures, word_to_features
import warnings

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", message="Recall is ill-defined")

input_file_name = sys.argv[1]
model_name = sys.argv[2]
feature_map_file = sys.argv[3]
out_file_name = sys.argv[4]
out_file = open(out_file_name, "w")


def check_availability(element, collection):
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


    # def process_line(line):
    #     return line.split()


    # def read_file(file):
    #     with open(file, 'r') as file:
    #         return [process_line(line) for line in file.readlines()]

def process_line(line):
    return [re.search(r'(.*)/.*$', pair).groups()[0] for pair in line.split()]

def read_input(file):
    try:
        with open(file, 'r') as f:
            sents = [process_line(line) for line in f.readlines()]
    except AttributeError:
        with open(file, 'r') as f:
            sents = [line.split() for line in f.readlines()]
    return sents

test_sents = read_input(input_file_name)

clf = load(model_name)

feature_mapping = load(feature_map_file)
possible_tags, vec, le = feature_mapping


def check_if_rare_with_vectorizer(word):
    if check_availability("w_i=" + word, vec.get_feature_names()):
        return False
    return True


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

            features = word_to_features(tok, check_if_rare_with_vectorizer, word_minus_2, word_minus_1, word_plus_1, word_plus_2, tag_minus_2_1, tag_minus_1)

            # greedy tagger, we take the best tag at each point
            tag = clf.predict(vec.transform(features).todense())[0]
            tag = le.inverse_transform([tag])[0]

            y_predicted.append(tag)
            predictions_for_line.append(tag)

            word_plus_1 = word_plus_2

            if it.has_next():
                word_plus_2 = it.next()
            else:
                word_plus_2 = END

            word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = word_minus_1, tag_minus_1 + "_" + tag, tag, tok

        for tok, tag in zip(sent, predictions_for_line):
            print(tok + "/" + tag, end=" ", file=out_file)
        print("", file=out_file)


greedy_predict(test_sents, out_file)
out_file.close()